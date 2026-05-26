# 2026-05-18 LIBERO Joint Finite-Difference MeanFlow Pilot Worklog

## Decision

- Keep the FastWAMJoint causal video/action topology.
- Use finite-difference MeanFlow for both video and action.
- Start from `checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
- Use 4 GPUs for the pilot. GPUs 4-7 are already occupied, so 8-card training is not the clean choice here.
- Set `save_training_state=true` so later continuation can resume from the state directory, not only from `.pt` weights.

## Archived Plan

- `docs/plan/2026-05-18-libero-joint-finite-difference-meanflow-pilot.md`

## Implemented Files

- `src/fastwam/models/wan22/fastwam_joint_meanflow.py`
- `src/fastwam/runtime.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/wan_video_dit.py`
- `configs/model/fastwam_joint_meanflow.yaml`
- `configs/task/libero_joint_meanflow_fd_2cam224_1e-4.yaml`

## Smoke Check

Smoke passed after fixing the bf16 dtype path in `WanVideoDiT.pre_dit`.

Saved checkpoint:

```text
runs/libero_joint_meanflow_fd_2cam224_1e-4/libero_joint_meanflow_fd_smoke2_20260518/checkpoints/weights/step_000001.pt
```

The smoke run loaded the release checkpoint, ran one training step, and wrote a weight checkpoint successfully.

## Pilot Run

Started with:

```text
PATH=/DATA/disk2/wangchen/projects/FastWAM/.conda/fastwam/bin:$PATH \
HF_HOME=/DATA/disk3/tmp/hf_home \
HF_DATASETS_CACHE=/DATA/disk3/tmp/hf_home/datasets \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29530 \
RUN_ID=libero_joint_meanflow_fd_pilot2k_20260518 \
bash scripts/train_zero2.sh 4 \
  task=libero_joint_meanflow_fd_2cam224_1e-4 \
  max_steps=2000 \
  save_every=500 \
  save_training_state=true \
  wandb.enabled=false
```

Logs:

```text
/DATA/disk3/tmp/fastwam_joint_meanflow_20260518/libero_joint_meanflow_fd_pilot2k_20260518.log
```

Expected outputs:

```text
runs/libero_joint_meanflow_fd_2cam224_1e-4/libero_joint_meanflow_fd_pilot2k_20260518/
```

Checkpointing:

- weight checkpoints every 500 steps
- full training state saved alongside weights

## Current Status

As of the latest log check, the pilot has reached:

```text
epoch=0 step=20/2000
```

No training-time error has appeared so far. The run is still active.
