import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _to_trace_action_array(action: Any) -> list[list[float]]:
    if isinstance(action, torch.Tensor):
        arr = action.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected action trace stage [T, D] or [1, T, D], got {tuple(arr.shape)}")
    return arr.astype(np.float32, copy=False).tolist()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]], dict[str, list[list[float]]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]
    action_stages = {
        "model_normalized": _to_trace_action_array(action),
    }

    action = _denormalize_action(action, processor)[0]  # [T, D]
    action_stages["denormalized_dataset"] = _to_trace_action_array(action)

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action_stages["gripper_scaled_before_invert"] = _to_trace_action_array(action)
    action = invert_gripper_action(action)
    action_stages["libero_continuous_before_sign"] = _to_trace_action_array(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    action_stages["env_action"] = _to_trace_action_array(action)
    return action, imgs, predicted_future_frames, action_stages


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _resolve_trial_indices(cfg: DictConfig) -> tuple[list[int], bool]:
    num_trials = int(cfg.EVALUATION.num_trials)
    trial_indices_cfg = cfg.EVALUATION.get("trial_indices", None)
    if trial_indices_cfg is None:
        return list(range(num_trials)), False

    if isinstance(trial_indices_cfg, str):
        value = trial_indices_cfg.strip()
        if value == "" or value.lower() in {"none", "null"}:
            return list(range(num_trials)), False
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        raw_indices = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(trial_indices_cfg, (list, tuple, ListConfig)):
        raw_indices = list(trial_indices_cfg)
    else:
        raw_indices = [trial_indices_cfg]

    trial_indices = [int(idx) for idx in raw_indices]
    if len(trial_indices) == 0:
        raise ValueError("EVALUATION.trial_indices was provided but no trial index was found.")
    if len(set(trial_indices)) != len(trial_indices):
        raise ValueError(f"EVALUATION.trial_indices contains duplicates: {trial_indices}")

    out_of_range = [idx for idx in trial_indices if idx < 0 or idx >= num_trials]
    if out_of_range:
        raise ValueError(
            f"EVALUATION.trial_indices must be within [0, {num_trials - 1}], "
            f"got out-of-range values: {out_of_range}."
        )
    return trial_indices, True


def _trial_indices_output_suffix(trial_indices: list[int], is_selected_rerun: bool) -> str:
    if not is_selected_rerun:
        return "_results.json"
    if len(trial_indices) <= 8:
        trial_tag = "-".join(str(idx) for idx in trial_indices)
    else:
        trial_tag = f"{trial_indices[0]}-{trial_indices[-1]}_n{len(trial_indices)}"
    return f"_trials{trial_tag}_video_rerun.json"


def _summarize_action_array(actions: list[list[float]]) -> dict[str, Any]:
    if len(actions) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
            "near_limit_fraction": None,
            "mean_abs_delta": None,
            "gripper_value_counts": {},
            "gripper_transitions": 0,
        }

    arr = np.asarray(actions, dtype=np.float32)
    gripper = arr[:, -1]
    rounded_gripper = np.round(gripper, decimals=3)
    values, counts = np.unique(rounded_gripper, return_counts=True)
    gripper_value_counts = {str(float(value)): int(count) for value, count in zip(values, counts)}
    gripper_transitions = int(np.sum(np.abs(np.diff(np.sign(gripper))) > 0)) if len(gripper) > 1 else 0
    mean_abs_delta = (
        np.mean(np.abs(np.diff(arr, axis=0)), axis=0).tolist() if arr.shape[0] > 1 else [0.0] * arr.shape[1]
    )

    return {
        "count": int(arr.shape[0]),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "abs_max": np.max(np.abs(arr), axis=0).tolist(),
        "near_limit_fraction": {
            "eef_pos_abs_ge_0_90": float(np.mean(np.abs(arr[:, :3]) >= 0.90)),
            "rot_abs_ge_0_35": float(np.mean(np.abs(arr[:, 3:6]) >= 0.35)),
        },
        "mean_abs_delta": mean_abs_delta,
        "gripper_value_counts": gripper_value_counts,
        "gripper_transitions": gripper_transitions,
    }


def _make_action_trace(
    *,
    task_description: str,
    episode_idx: int,
    action_horizon: int,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    use_action_ensembler: bool,
    binarize_gripper: bool,
    save_raw_action_trace: bool,
) -> dict[str, Any]:
    return {
        "task_description": task_description,
        "episode_idx": int(episode_idx),
        "action_dimensions": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "action_space_reference": {
            "source": "release dataset stats after denormalization",
            "eef_pos_approx_range": [-0.9375, 0.9375],
            "rot_approx_range": {
                "droll": [-0.242, 0.356],
                "dpitch": [-0.375, 0.375],
                "dyaw": [-0.364, 0.375],
            },
            "gripper_after_eval": "binarized/sign action sent to LIBERO env",
        },
        "config": {
            "action_horizon": int(action_horizon),
            "max_steps": int(max_steps),
            "num_steps_wait": int(num_steps_wait),
            "replan_steps": int(replan_steps),
            "use_action_ensembler": bool(use_action_ensembler),
            "binarize_gripper": bool(binarize_gripper),
            "save_raw_action_trace": bool(save_raw_action_trace),
        },
        "dummy_wait_actions": [],
        "replans": [],
        "executed_policy_actions": [],
        "summary": None,
    }


def _write_action_trace(
    action_trace_dir: Path,
    cfg: DictConfig,
    episode_idx: int,
    success: bool,
    trace: dict[str, Any],
) -> None:
    action_trace_dir.mkdir(parents=True, exist_ok=True)
    trace = dict(trace)
    trace["success"] = bool(success)
    trace["summary"] = _summarize_action_array(
        [record["action"] for record in trace.get("executed_policy_actions", [])]
    )
    output_file = action_trace_dir / (
        f"task{cfg.EVALUATION.task_id}_trial{episode_idx}_success{bool(success)}_action_trace.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, cls=NumpyEncoder)


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], Optional[dict[str, Any]]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    save_action_trace = bool(cfg.EVALUATION.get("save_action_trace", False))
    save_raw_action_trace = bool(cfg.EVALUATION.get("save_raw_action_trace", save_action_trace))
    diagnose_action_values = bool(cfg.EVALUATION.get("diagnose_action_values", False))
    action_abs_limit_cfg = cfg.EVALUATION.get("diagnose_action_abs_limit", None)
    action_abs_limit = float(action_abs_limit_cfg) if action_abs_limit_cfg is not None else None
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])
    action_trace = None
    if save_action_trace:
        action_trace = _make_action_trace(
            task_description=task_description,
            episode_idx=episode_idx,
            action_horizon=action_horizon,
            max_steps=max_steps,
            num_steps_wait=num_steps_wait,
            replan_steps=replan_steps,
            use_action_ensembler=use_action_ensembler,
            binarize_gripper=bool(cfg.EVALUATION.get("binarize_gripper", False)),
            save_raw_action_trace=save_raw_action_trace,
        )

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            dummy_action = get_libero_dummy_action()
            if action_trace is not None:
                action_trace["dummy_wait_actions"].append({"env_t": int(t), "action": list(dummy_action)})
            obs, _, done, _ = env.step(dummy_action)
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames, action_stages = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            if action_trace is not None:
                chunk_actions = action_chunk.tolist()
                replan_record = {
                    "replan_idx": int(len(action_trace["replans"])),
                    "env_t": int(t),
                    "chunk_shape": list(action_chunk.shape),
                    "executed_count": int(len(pending_actions)),
                    "chunk_summary": _summarize_action_array(chunk_actions),
                    "chunk_actions": chunk_actions,
                }
                if save_raw_action_trace:
                    replan_record["chunk_actions_by_stage"] = action_stages
                    replan_record["chunk_stage_summaries"] = {
                        stage_name: _summarize_action_array(stage_actions)
                        for stage_name, stage_actions in action_stages.items()
                    }
                action_trace["replans"].append(replan_record)
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        action_idx_in_replan = int(replan_steps - len(pending_actions))
        executed_action = pending_actions.pop(0)
        if diagnose_action_values:
            action_arr = np.asarray(executed_action, dtype=np.float32)
            finite = bool(np.all(np.isfinite(action_arr)))
            max_abs = float(np.max(np.abs(action_arr))) if action_arr.size > 0 else 0.0
            invalid_reason = None
            if not finite:
                invalid_reason = "nonfinite"
            elif action_abs_limit is not None and max_abs > action_abs_limit:
                invalid_reason = f"abs>{action_abs_limit:g}"
            if invalid_reason is not None:
                diagnostic_record = {
                    "env_t": int(t),
                    "replan_idx": int(len(action_trace["replans"]) - 1) if action_trace is not None else -1,
                    "action_idx_in_replan": int(action_idx_in_replan),
                    "reason": invalid_reason,
                    "finite": finite,
                    "max_abs": max_abs,
                    "action": action_arr.tolist(),
                }
                logging.error("Invalid eval action detected: %s", diagnostic_record)
                if action_trace is not None:
                    action_trace.setdefault("diagnostics", []).append(diagnostic_record)
                    action_trace["invalid_action_failure"] = diagnostic_record
                pbar.close()
                return False, replay_images, predicted_future_video_clips, None, action_trace
        if action_trace is not None:
            executed_record = {
                "env_t": int(t),
                "replan_idx": int(len(action_trace["replans"]) - 1),
                "action_idx_in_replan": action_idx_in_replan,
                "action": list(executed_action),
            }
            if save_raw_action_trace and action_trace["replans"]:
                raw_stages = action_trace["replans"][-1].get("chunk_actions_by_stage", {})
                executed_record["action_by_stage"] = {
                    stage_name: stage_actions[action_idx_in_replan]
                    for stage_name, stage_actions in raw_stages.items()
                    if action_idx_in_replan < len(stage_actions)
                }
            action_trace["executed_policy_actions"].append(executed_record)
        obs, _, done, _ = env.step(executed_action)
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, action_trace


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    action_trace_dir: Path,
    *,
    trial_indices: list[int],
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "evaluated_trials": trial_indices,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    save_action_trace = bool(cfg.EVALUATION.get("save_action_trace", False))

    for trial_idx in trial_indices:
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, action_trace = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        if bool(cfg.EVALUATION.get("save_rollout_video", True)):
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
        if save_action_trace and action_trace is not None:
            _write_action_trace(
                action_trace_dir=action_trace_dir,
                cfg=cfg,
                episode_idx=trial_idx,
                success=success,
                trace=action_trace,
            )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)
    action_trace_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "action_traces"
    if bool(cfg.EVALUATION.get("save_action_trace", False)):
        action_trace_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)
    trial_indices, is_selected_rerun = _resolve_trial_indices(cfg)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": len(trial_indices),
        "configured_num_trials": int(cfg.EVALUATION.num_trials),
        "selected_trial_rerun": is_selected_rerun,
        "evaluated_trials": trial_indices,
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_trace_dir=action_trace_dir,
        trial_indices=trial_indices,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_suffix = _trial_indices_output_suffix(trial_indices, is_selected_rerun)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}{result_suffix}"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{len(trial_indices)} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
