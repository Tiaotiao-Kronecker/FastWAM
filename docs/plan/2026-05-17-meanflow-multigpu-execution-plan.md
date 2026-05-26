# 2026-05-17 MeanFlow Multigpu Execution Plan

Purpose: run the next MeanFlow checks without mixing output locations or changing
too many training variables at once.

## Current GPU Policy

Use two 4-GPU training jobs instead of one 8-GPU job.

Reason:

```text
batch_size=1 per process for the LIBERO MeanFlow task.
4 GPUs => global batch 4, matching the finished finite-difference 2k pilot.
8 GPUs => global batch 8, which changes optimization while testing training length.
```

Default allocation:

```text
paper-JVP C-action-only pilot:        GPUs 0,2,3,4
finite-difference continuation:       GPUs 1,5,6,7
small LIBERO gap-probe eval:          use whichever training group is idle first
```

If GPU 1 shows sustained non-FastWAM compute load, move the continuation group to
`4,5,6,7` and move JVP to `0,2,3,6`, keeping groups disjoint.

## ZeRO-2 Meaning

`scripts/train_zero2.sh` launches Accelerate with DeepSpeed ZeRO stage 2:

```text
Accelerate config: scripts/accelerate_configs/accelerate_zero2_ds.yaml
DeepSpeed config: scripts/ds_configs/ds_zero2_config.json
```

ZeRO-2 shards optimizer states and gradients across GPUs. Model parameters are
still replicated on each GPU. It saves much more memory than plain DDP / ZeRO-1,
but it is less aggressive than ZeRO-3. This repo uses no optimizer or parameter
CPU offload in the ZeRO-2 config.

## Canonical Output Roots

Training root:

```text
runs/libero_one_step_meanflow_2cam224_1e-4/<RUN_ID>
```

Training files under each run:

```text
config.yaml
dataset_stats.json
checkpoints/weights/step_*.pt
checkpoints/state/step_*/          only when save_training_state=true
eval/                              train-time eval artifacts, normally unused here
```

External logs:

```text
/DATA/disk3/tmp/fastwam_meanflow_20260517/<RUN_ID>.log
```

Evaluation root:

```text
evaluate_results/libero/<EVAL_RUN_ID>
```

Evaluation files:

```text
summary.csv
<suite>/gpu*_task*_results.json
<suite>/videos/*.mp4               only when SAVE_ROLLOUT_VIDEO=true
<suite>/action_traces/*_action_trace.json
task_logs/
task_status/
```

Trace summaries:

```text
evaluate_results/libero/<EVAL_RUN_ID>_trace_summary/
```

## Training Visualization

Local visualization script:

```text
experiments/libero/visualize_meanflow_training.py
```

Default output:

```text
evaluate_results/training/meanflow_20260517/index.html
evaluate_results/training/meanflow_20260517/training_metrics.csv
evaluate_results/training/meanflow_20260517/training_summary.json
```

Refresh command while the two training jobs are running:

```bash
python experiments/libero/visualize_meanflow_training.py \
  --log-dir /DATA/disk3/tmp/fastwam_meanflow_20260517 \
  --output-dir evaluate_results/training/meanflow_20260517 \
  --title "MeanFlow 2026-05-17 Training Dashboard"
```

Open:

```text
evaluate_results/training/meanflow_20260517/index.html
```

How to read:

```text
Total Loss / MeanFlow Target Loss:
  Should trend down, but random r/t means local spikes are normal.

Training Speed:
  Should stabilize after warmup. A sudden drop usually means GPU contention,
  dataloader stall, or a stuck distributed worker.

Learning Rate:
  Confirms whether a run is a full-state resume or a weight-only restart.

MeanFlow Interval / Sigma Start / Sigma End:
  For random r/t, these should stay spread out. Interval collapsing near zero
  means the objective is not exercising the intended range.

Decision rule:
  Loss decrease is only a sanity signal. Promotion still requires gap-probe
  rollout improvement and release-like gripper/action-trace timing.
```

Initial parser check was generated from old logs at:

```text
evaluate_results/training/meanflow_20260517_initial/index.html
```

## Existing Baseline Inputs

Release checkpoint:

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

Release dataset stats:

```text
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

Finished finite-difference 2k checkpoint:

```text
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_2k_20260516/checkpoints/weights/step_002000.pt
```

Important caveat:

```text
The finished 2k run used save_training_state=false.
Continuation from step_002000.pt is therefore weight-only.
Optimizer, scheduler, dataloader position, and trainer global_step are reset.
New continuation checkpoint tags count additional steps, not absolute total steps.
```

Example:

```text
new continuation step_004000.pt == effective 2k + 4k = 6k training
new continuation step_006000.pt == effective 2k + 6k = 8k training
```

## Experiment A: Paper-JVP C-Action-Only Pilot

RUN_ID:

```text
meanflow_c_action_paper_jvp_fp32_rt025_2k_20260517
```

Goal:

```text
Test the paper-style objective: JVP derivative, random r/t, equal-time probability 0.25.
Keep the same C-action-only trainable scope as the finite-difference pilot.
```

Training command:

```bash
mkdir -p /DATA/disk3/tmp/fastwam_meanflow_20260517

HF_HOME=/DATA/disk3/tmp/hf_home \
HF_DATASETS_CACHE=/DATA/disk3/tmp/hf_home/datasets \
CUDA_VISIBLE_DEVICES=0,2,3,4 \
MASTER_PORT=29517 \
RUN_ID=meanflow_c_action_paper_jvp_fp32_rt025_2k_20260517 \
bash scripts/train_zero2.sh 4 \
  task=libero_one_step_meanflow_2cam224_1e-4 \
  model.one_step_meanflow.objective=paper_jvp \
  model.one_step_meanflow.random_timesteps=true \
  model.one_step_meanflow.equal_time_prob=0.25 \
  mixed_precision=no \
  max_steps=2000 \
  save_every=500 \
  save_training_state=true \
  wandb.enabled=false \
  > /DATA/disk3/tmp/fastwam_meanflow_20260517/meanflow_c_action_paper_jvp_fp32_rt025_2k_20260517.log 2>&1
```

Checkpoints to evaluate:

```text
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_paper_jvp_fp32_rt025_2k_20260517/checkpoints/weights/step_000500.pt
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_paper_jvp_fp32_rt025_2k_20260517/checkpoints/weights/step_001000.pt
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_paper_jvp_fp32_rt025_2k_20260517/checkpoints/weights/step_002000.pt
```

## Experiment B: Finite-Difference Continuation

RUN_ID:

```text
meanflow_c_action_fd_bf16_0123_from2k_plus6k_20260517
```

Goal:

```text
Test whether the finite-difference failure was mainly undertraining.
This is not an exact optimizer-state continuation; it is a weight-only continuation.
```

Training command:

```bash
mkdir -p /DATA/disk3/tmp/fastwam_meanflow_20260517

HF_HOME=/DATA/disk3/tmp/hf_home \
HF_DATASETS_CACHE=/DATA/disk3/tmp/hf_home/datasets \
CUDA_VISIBLE_DEVICES=1,5,6,7 \
MASTER_PORT=29527 \
RUN_ID=meanflow_c_action_fd_bf16_0123_from2k_plus6k_20260517 \
bash scripts/train_zero2.sh 4 \
  task=libero_one_step_meanflow_2cam224_1e-4 \
  model.one_step_meanflow.objective=finite_difference \
  model.one_step_meanflow.random_timesteps=true \
  model.one_step_meanflow.equal_time_prob=0.0 \
  mixed_precision=bf16 \
  resume=./runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_2k_20260516/checkpoints/weights/step_002000.pt \
  max_steps=6000 \
  save_every=2000 \
  save_training_state=true \
  wandb.enabled=false \
  > /DATA/disk3/tmp/fastwam_meanflow_20260517/meanflow_c_action_fd_bf16_0123_from2k_plus6k_20260517.log 2>&1
```

Effective checkpoints:

```text
step_002000.pt => effective 4k
step_004000.pt => effective 6k
step_006000.pt => effective 8k
```

Checkpoint paths:

```text
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_from2k_plus6k_20260517/checkpoints/weights/step_002000.pt
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_from2k_plus6k_20260517/checkpoints/weights/step_004000.pt
runs/libero_one_step_meanflow_2cam224_1e-4/meanflow_c_action_fd_bf16_0123_from2k_plus6k_20260517/checkpoints/weights/step_006000.pt
```

## Canonical Gap-Probe Eval

Task file:

```text
experiments/libero/task_sets/libero_gap_probe_v1.txt
```

Run all probes with:

```text
NUM_INFERENCE_STEPS=1
NUM_TRIALS=1
TRIAL_INDICES=[0]
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false
WORKERS_PER_GPU=1
```

JVP eval IDs:

```text
meanflow_c_jvp_fp32_rt025_step0500_gap_probe_t0_20260517
meanflow_c_jvp_fp32_rt025_step1000_gap_probe_t0_20260517
meanflow_c_jvp_fp32_rt025_step2000_gap_probe_t0_20260517
```

Finite-difference continuation eval IDs:

```text
meanflow_c_fd_bf16_0123_eff4k_gap_probe_t0_20260517
meanflow_c_fd_bf16_0123_eff6k_gap_probe_t0_20260517
meanflow_c_fd_bf16_0123_eff8k_gap_probe_t0_20260517
```

Gate:

```text
Promote only if gap-probe rollout improves and gripper segments become
release-like. Loss decrease without rollout/gripper improvement is not enough.
```
