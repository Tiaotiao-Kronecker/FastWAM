import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402


register_default_resolvers()
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FastWAM release pure inference latency for different action denoise steps."
    )
    parser.add_argument("--ckpt", default="./checkpoints/fastwam_release/libero_uncond_2cam224.pt")
    parser.add_argument("--config-name", default="sim_libero.yaml")
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--output", default="./evaluate_results/latency/release_1_vs_10_worker.json")
    parser.add_argument("--image-mode", choices=["zeros", "random"], default="random")
    parser.add_argument("--context-mode", choices=["zeros", "random"], default="random")
    parser.add_argument("--rand-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--measure-loop", action="store_true", default=True)
    parser.add_argument("--no-measure-loop", dest="measure_loop", action="store_false")
    return parser.parse_args()


def _load_cfg(args: argparse.Namespace) -> Any:
    config_dir = str(PROJECT_ROOT / "configs")
    overrides = [
        f"task={args.task}",
        f"ckpt={args.ckpt}",
        f"gpu_id={args.gpu_id}",
        "model.load_text_encoder=false",
        "model.skip_dit_load_from_pretrain=true",
        "model.action_dit_pretrained_path=null",
        f"EVALUATION.device=cuda:{args.gpu_id}",
        "EVALUATION.save_rollout_video=false",
        "EVALUATION.visualize_future_video=false",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return hydra.compose(config_name=args.config_name, overrides=overrides)


def _dtype_from_precision(precision: str) -> torch.dtype:
    key = str(precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {precision}")


def _load_model(cfg: Any, device: str) -> torch.nn.Module:
    model_dtype = _dtype_from_precision(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(cfg.ckpt))
    return model.to(device).eval()


def _stats(values_s: list[float]) -> dict[str, float]:
    values_ms = sorted(v * 1000.0 for v in values_s)
    if not values_ms:
        raise ValueError("No timing values to summarize.")

    def percentile(p: float) -> float:
        if len(values_ms) == 1:
            return values_ms[0]
        rank = (len(values_ms) - 1) * p
        lo = int(rank)
        hi = min(lo + 1, len(values_ms) - 1)
        weight = rank - lo
        return values_ms[lo] * (1.0 - weight) + values_ms[hi] * weight

    return {
        "mean_ms": float(statistics.fmean(values_ms)),
        "median_ms": float(percentile(0.50)),
        "p10_ms": float(percentile(0.10)),
        "p90_ms": float(percentile(0.90)),
        "min_ms": float(values_ms[0]),
        "max_ms": float(values_ms[-1]),
        "std_ms": float(statistics.pstdev(values_ms)) if len(values_ms) > 1 else 0.0,
    }


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _make_inputs(cfg: Any, model: torch.nn.Module, args: argparse.Namespace, device: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    video_size = cfg.data.train.get("video_size", [224, 448])
    height, width = int(video_size[0]), int(video_size[1])
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    context_len = int(cfg.data.train.get("context_len", cfg.model.get("tokenizer_max_len", 128)))
    text_dim = int(cfg.model.action_dit_config.text_dim)

    image_shape = (1, 3, height, width)
    context_shape = (1, context_len, text_dim)
    if args.image_mode == "zeros":
        input_image = torch.zeros(image_shape, dtype=torch.float32)
    else:
        input_image = torch.rand(image_shape, dtype=torch.float32)

    if args.context_mode == "zeros":
        context = torch.zeros(context_shape, dtype=torch.float32)
    else:
        context = torch.randn(context_shape, dtype=torch.float32)
    context_mask = torch.ones((1, context_len), dtype=torch.bool)
    proprio = torch.zeros((1, proprio_dim), dtype=torch.float32)

    return {
        "input_image": input_image,
        "context": context,
        "context_mask": context_mask,
        "proprio": proprio,
        "action_horizon": torch.tensor(int(cfg.data.train.num_frames) - 1),
    }


@torch.inference_mode()
def _prepare_denoise_cache(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    device: str,
) -> dict[str, Any]:
    input_image = inputs["input_image"].to(device=device, dtype=model.torch_dtype)
    context = inputs["context"].to(device=device, dtype=model.torch_dtype)
    context_mask = inputs["context_mask"].to(device=device, dtype=torch.bool)
    proprio = inputs["proprio"].to(device=device, dtype=model.torch_dtype)

    first_frame_latents = model._encode_input_image_latents_tensor(input_image=input_image, tiled=False)
    if proprio is not None:
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

    timestep_video = torch.zeros(
        (first_frame_latents.shape[0],),
        dtype=first_frame_latents.dtype,
        device=device,
    )
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    action_horizon = int(inputs["action_horizon"].item())
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_horizon,
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    video_kv_cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    return {
        "context": context,
        "context_mask": context_mask,
        "video_kv_cache": video_kv_cache,
        "attention_mask": attention_mask,
        "video_seq_len": video_seq_len,
        "action_horizon": action_horizon,
    }


@torch.inference_mode()
def _run_denoise_loop_once(
    model: torch.nn.Module,
    cache: dict[str, Any],
    *,
    num_inference_steps: int,
    device: str,
) -> torch.Tensor:
    latents_action = torch.randn(
        (1, int(cache["action_horizon"]), model.action_expert.action_dim),
        device=device,
        dtype=model.torch_dtype,
    )
    infer_timesteps_action, infer_deltas_action = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=int(num_inference_steps),
        device=torch.device(device),
        dtype=latents_action.dtype,
        shift_override=None,
    )
    for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
        timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=device)
        pred_action = model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=cache["context"],
            context_mask=cache["context_mask"],
            video_kv_cache=cache["video_kv_cache"],
            attention_mask=cache["attention_mask"],
            video_seq_len=cache["video_seq_len"],
        )
        latents_action = model.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
    return latents_action


def _benchmark_step(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    cache: dict[str, Any],
    args: argparse.Namespace,
    *,
    step: int,
    device: str,
) -> dict[str, Any]:
    end_to_end: list[float] = []
    denoise_loop: list[float] = []

    for idx in range(args.warmup + args.repeats):
        _synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model.infer_action(
                prompt=None,
                input_image=inputs["input_image"],
                action_horizon=int(inputs["action_horizon"].item()),
                proprio=inputs["proprio"],
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                negative_prompt="",
                text_cfg_scale=1.0,
                num_inference_steps=int(step),
                sigma_shift=None,
                seed=None,
                rand_device=args.rand_device,
                tiled=False,
            )
        _synchronize(device)
        elapsed = time.perf_counter() - start
        if idx >= args.warmup:
            end_to_end.append(elapsed)

    if args.measure_loop:
        for idx in range(args.warmup + args.repeats):
            _synchronize(device)
            start = time.perf_counter()
            _ = _run_denoise_loop_once(model, cache, num_inference_steps=step, device=device)
            _synchronize(device)
            elapsed = time.perf_counter() - start
            if idx >= args.warmup:
                denoise_loop.append(elapsed)

    return {
        "num_inference_steps": int(step),
        "end_to_end": _stats(end_to_end),
        "denoise_loop": _stats(denoise_loop) if denoise_loop else None,
        "raw_end_to_end_s": end_to_end,
        "raw_denoise_loop_s": denoise_loop,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"

    cfg = _load_cfg(args)
    model = _load_model(cfg, device=device)
    inputs = _make_inputs(cfg, model, args, device=device)
    with torch.inference_mode():
        cache = _prepare_denoise_cache(model, inputs, device=device)
    _synchronize(device)

    results = []
    for step in args.steps:
        print(f"[latency] gpu={args.gpu_id} steps={step} warmup={args.warmup} repeats={args.repeats}", flush=True)
        results.append(_benchmark_step(model, inputs, cache, args, step=step, device=device))

    by_step = {item["num_inference_steps"]: item for item in results}
    ratios: dict[str, float] = {}
    if 1 in by_step and 10 in by_step:
        ratios["end_to_end_mean_10_over_1"] = (
            by_step[10]["end_to_end"]["mean_ms"] / by_step[1]["end_to_end"]["mean_ms"]
        )
        if by_step[10]["denoise_loop"] is not None and by_step[1]["denoise_loop"] is not None:
            ratios["denoise_loop_mean_10_over_1"] = (
                by_step[10]["denoise_loop"]["mean_ms"] / by_step[1]["denoise_loop"]["mean_ms"]
            )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "gpu_id": args.gpu_id,
        "gpu_name": torch.cuda.get_device_name(args.gpu_id),
        "ckpt": str(cfg.ckpt),
        "task": args.task,
        "config_name": args.config_name,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "image_mode": args.image_mode,
        "context_mode": args.context_mode,
        "rand_device": args.rand_device,
        "torch_version": torch.__version__,
        "input": {
            "image_shape": list(inputs["input_image"].shape),
            "context_shape": list(inputs["context"].shape),
            "proprio_shape": list(inputs["proprio"].shape),
            "action_horizon": int(inputs["action_horizon"].item()),
        },
        "results": results,
        "ratios": ratios,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["created_at", "gpu_id", "gpu_name", "ratios"]}, indent=2), flush=True)
    print(f"[latency] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
