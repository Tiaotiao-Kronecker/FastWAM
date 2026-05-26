import argparse
import os
import time
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging
from fastwam.utils.pytorch_utils import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["paper_jvp", "finite_difference"], required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def format_metrics(step: int, loss: torch.Tensor, metrics: dict[str, float]) -> str:
    fields = [f"diag_step={step}", f"loss={float(loss.detach().float().item()):.6g}"]
    for key in sorted(metrics):
        fields.append(f"{key}={float(metrics[key]):.6g}")
    return " ".join(fields)


def gather_metrics(trainer: Wan22Trainer, loss: torch.Tensor, metrics: dict[str, float]) -> tuple[float, dict[str, float]]:
    global_loss = float(trainer.accelerator.gather(loss.detach().float().reshape(1)).mean().item())
    global_metrics = {}
    for key, value in metrics.items():
        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
        global_metrics[key] = float(trainer.accelerator.gather(metric_tensor).mean().item())
    return global_loss, global_metrics


def run_train_diag(cfg, args, model_dtype: torch.dtype):
    model_device = _resolve_train_device()
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)
    trainer = Wan22Trainer(cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds)

    data_iter = iter(trainer.train_loader)
    trainer.run_start_step = trainer.global_step
    trainer.run_start_time = time.perf_counter()

    for _ in range(int(args.steps)):
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.train_loader)
            sample = next(data_iter)

        with trainer.accelerator.accumulate(trainer.model):
            train_model = (
                trainer.model
                if hasattr(trainer.model, "training_loss")
                else trainer.accelerator.unwrap_model(trainer.model)
            )
            with trainer.accelerator.autocast():
                loss, metrics = train_model.training_loss(sample)
            trainer.accelerator.backward(loss)

            if trainer.accelerator.sync_gradients:
                trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
                trainer.optimizer.step()
                if not trainer.accelerator.optimizer_step_was_skipped:
                    trainer.scheduler.step()
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer.global_step += 1

                global_loss, global_metrics = gather_metrics(trainer, loss, metrics)
                if trainer.accelerator.is_main_process:
                    detached_loss = torch.tensor(global_loss)
                    print(format_metrics(trainer.global_step, detached_loss, global_metrics), flush=True)

        del loss, metrics, sample

    trainer.accelerator.wait_for_everyone()
    trainer._finish_wandb()


def run_forward_diag(cfg, args, model_dtype: torch.dtype):
    print(f"[diag] resume={cfg.resume}")
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    if cfg.resume:
        model.load_checkpoint(str(cfg.resume), optimizer=None)
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)

    train_ds = instantiate(cfg.data.train)
    loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers))

    for step, sample in zip(range(1, int(args.steps) + 1), loader):
        loss, metrics = model.training_loss(sample)
        print(format_metrics(step, loss, metrics), flush=True)
        del loss, metrics, sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    register_default_resolvers()
    os.environ.setdefault("FASTWAM_MEANFLOW_DEBUG_NORMS", "1")
    setup_logging()

    overrides = [
        "task=libero_c_joint_meanflow_2cam224_1e-4",
        f"model.c_joint_meanflow.objective={args.objective}",
        "batch_size=1",
        "num_workers=0",
        "mixed_precision=no",
        "wandb.enabled=false",
        f"output_dir=/DATA/disk3/tmp/fastwam_c_joint_diag/{args.objective}",
        *args.overrides,
    ]

    with hydra.initialize(config_path="../configs", version_base="1.3"):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)

    set_global_seed(int(cfg.seed))
    misc.register_work_dir(cfg.output_dir)
    model_dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))

    mode = "train" if args.train else "forward"
    print(f"[diag] objective={args.objective} mode={mode} steps={args.steps} device={args.device} dtype={model_dtype}")
    if args.train:
        run_train_diag(cfg, args, model_dtype)
    else:
        run_forward_diag(cfg, args, model_dtype)


if __name__ == "__main__":
    main()
