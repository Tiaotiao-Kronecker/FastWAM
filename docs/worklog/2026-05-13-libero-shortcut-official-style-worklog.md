# 2026-05-13 LIBERO Official-Style Shortcut Worklog

## Motivation

The previous `shortcut_1_2k` implementation was a simplified one-step objective. After checking the Shortcut Models paper and official JAX code, the main gaps were:

- It fixed the training start at sigma=1 and step size d=1.
- It used `step_size * num_train_timesteps` as conditioning instead of the paper-style `dt_base`.
- It trained direct endpoint/velocity and half-step losses, not the paper's mixed flow/bootstrap target.
- Its new shortcut conditioner lived under `action_expert`, but the trainer optimizer only collected `model.dit` parameters, so the conditioner was not included in optimization.

## Implemented Changes

Added a new official-style variant instead of overwriting previous results:

- `src/fastwam/models/wan22/fastwam_one_step_shortcut_official.py`
- `configs/model/fastwam_one_step_shortcut_official.yaml`
- `configs/task/libero_one_step_shortcut_official_2cam224_1e-4.yaml`
- `experiments/libero/run_shortcut_official_pilot_chain.sh`

The new objective:

- Uses `num_denoise_steps=128`, so `dt_base` ranges over powers of two.
- Uses flow grounding at smallest step size, with target `noise - action`.
- Uses bootstrap shortcut target for larger steps:
  `v_target = 0.5 * (v_half_1 + v_half_2)` with stop-gradient half-step predictions.
- Uses FastWAM's sigma convention: sigma=1 is noise, sigma=0 is action, so updates are `x_{sigma-d} = x_sigma - d * v`.
- Uses `dt_base` embedding for shortcut conditioning.
- Keeps one-step eval conditioning at `step_size=1.0`, which maps to `dt_base=0`.

Trainer fix:

- `Wan22Trainer` now asks a model for `extra_trainable_parameters()`.
- `FastWAMOneStepShortcut` exposes the shortcut conditioner parameters through that hook.

## Smoke Test

Command:

```bash
env CUDA_VISIBLE_DEVICES=4 .conda/fastwam/bin/python scripts/train.py \
  task=libero_one_step_shortcut_official_2cam224_1e-4 \
  output_dir=./runs/libero_one_step_shortcut_official_smoke_20260513 \
  max_steps=1 save_every=1 log_every=1 wandb.enabled=false
```

Result:

- Passed.
- Saved `runs/libero_one_step_shortcut_official_smoke_20260513/checkpoints/weights/step_000001.pt`.

## Current Experiment

Started:

```bash
tmux new-session -d -s shortcut_official_chain_20260513 \
  -c /DATA/disk2/wangchen/projects/FastWAM \
  "TRAIN_GPU=4 EVAL_GPUS=4 WORKERS_PER_GPU=1 experiments/libero/run_shortcut_official_pilot_chain.sh"
```

Training:

- Output: `runs/libero_one_step_shortcut_official_pilot2k_20260513`
- Expected checkpoint: `runs/libero_one_step_shortcut_official_pilot2k_20260513/checkpoints/weights/step_002000.pt`
- Initial progress: step 100/2000, about 0.88 step/s, ETA about 36 minutes.

Evaluation after training:

- Output: `evaluate_results/libero/shortcut_official_1_pilot2k_20260513`
- Checkpoint: `step_002000.pt`
- `num_inference_steps=1`
- `num_trials=50`
- `save_rollout_video=false`
- `EVAL_GPUS=4`
- `WORKERS_PER_GPU=1`
