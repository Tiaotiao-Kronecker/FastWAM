# 2026-05-29 RoboTwin Release10 Clean20 Follow-Up

## Purpose

Archive the `release_10` follow-up quick evaluation requested after the
MeanFlow joint-delta clean20 run. The question was whether the paired
joint-delta MeanFlow result is close to the true multi-step release behavior,
or whether it mostly tracks the one-step release baseline.

This is still a quick diagnostic: clean phase only, 20 online episodes per
task. It is useful for model-direction triage, not as a final formal benchmark.

## Run Metadata

Release10 follow-up run:

```text
evaluate_results/robotwin_quick_one_step_compare/20260529_release10_50task_clean20
```

Paired baseline run used for comparison:

```text
evaluate_results/robotwin_quick_one_step_compare/20260527_meanflow_jointdelta_50task_clean20
```

Timing from `manager.log`:

```text
start:  2026-05-29 10:59:03
finish: 2026-05-29 16:28:33
```

Scope:

```text
groups: release_10
phase: clean -> demo_clean
tasks: 50
episodes/task: 20
num_inference_steps: 10
gpu_ids: 6, 7
max_tasks_per_gpu: 1
total jobs: 50
total online rollouts: 50 tasks * 20 episodes = 1000
```

Completion checks:

```text
summary.csv: 51 lines = 50 result records + header
failed_jobs.txt: 0 bytes
manager status: quick compare finished
```

Primary result files:

```text
evaluate_results/robotwin_quick_one_step_compare/20260529_release10_50task_clean20/summary.csv
evaluate_results/robotwin_quick_one_step_compare/20260529_release10_50task_clean20/summary.json
evaluate_results/robotwin_quick_one_step_compare/20260529_release10_50task_clean20/manager.log
evaluate_results/robotwin_quick_one_step_compare/20260529_release10_50task_clean20/failed_jobs.txt
```

The result files live under `evaluate_results/`, which is intentionally ignored
by git. This document records the immutable run ids, paths, metrics, and
interpretation for source-control history.

## Compared Checkpoints

Release baseline:

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

MeanFlow joint-delta conditioner:

```text
runs/robotwin_one_step_meanflow_joint_conditioner_3cam_384_1e-4/joint_cond_fd_2k_20260525_prefix2200/checkpoints/weights/step_002000.pt
```

The baseline run already archived that the MeanFlow run used matching RoboTwin
dataset normalization/counts:

```text
num_episodes: 27,500
num_transition: 6,075,103
```

## Metric

Each task records:

```text
task_success_rate = successful rollouts / 20
```

The group-level value below is:

```text
group_mean = mean(task_success_rate over the same 50 completed tasks)
```

Because every task has exactly 20 episodes, this is also the episode-weighted
success rate over 1000 rollouts per group.

## Aggregate Same-Task Result

| Group | Completed Tasks | Rollouts | Successes | Mean Success |
| --- | ---: | ---: | ---: | ---: |
| `release_1` | 50 / 50 | 1000 | 722 | 0.7220 |
| `meanflow_1` joint-delta | 50 / 50 | 1000 | 740 | 0.7400 |
| `release_10` | 50 / 50 | 1000 | 923 | 0.9230 |

Deltas:

| Comparison | Delta |
| --- | ---: |
| `meanflow_1 - release_1` | +0.0180 |
| `release_10 - release_1` | +0.2010 |
| `release_10 - meanflow_1` | +0.1830 |

MeanFlow captures only about `0.018 / 0.201 = 8.96%` of the
`release_1 -> release_10` mean-success gap on this clean20 set.

## Task-Level Shape

Comparing `release_10` against `meanflow_1` on the same 50 tasks:

```text
release_10 > meanflow_1: 30 tasks
release_10 = meanflow_1: 17 tasks
release_10 < meanflow_1:  3 tasks
```

Mean absolute distance from MeanFlow:

```text
abs(meanflow_1 - release_1):  0.1080
abs(meanflow_1 - release_10): 0.1890
```

Per-task nearest-neighbor count:

```text
meanflow_1 closer to release_1:  20 tasks
meanflow_1 closer to release_10: 15 tasks
tie:                             15 tasks
```

Largest `release_10 - meanflow_1` gaps:

| Task | `release_1` | `meanflow_1` | `release_10` | `release_10 - meanflow_1` |
| --- | ---: | ---: | ---: | ---: |
| `handover_block` | 0.10 | 0.05 | 0.95 | +0.90 |
| `put_bottles_dustbin` | 0.10 | 0.10 | 0.90 | +0.80 |
| `handover_mic` | 0.85 | 0.35 | 1.00 | +0.65 |
| `move_stapler_pad` | 0.30 | 0.30 | 0.90 | +0.60 |
| `stack_blocks_three` | 0.45 | 0.35 | 0.95 | +0.60 |
| `place_cans_plasticbox` | 0.45 | 0.45 | 1.00 | +0.55 |
| `open_microwave` | 0.05 | 0.10 | 0.60 | +0.50 |
| `hanging_mug` | 0.35 | 0.20 | 0.70 | +0.50 |
| `place_dual_shoes` | 0.35 | 0.45 | 0.90 | +0.45 |
| `place_fan` | 0.30 | 0.55 | 1.00 | +0.45 |

Tasks where MeanFlow slightly exceeds `release_10`:

| Task | `release_1` | `meanflow_1` | `release_10` | `meanflow_1 - release_10` |
| --- | ---: | ---: | ---: | ---: |
| `place_phone_stand` | 0.80 | 1.00 | 0.95 | +0.05 |
| `stack_bowls_two` | 0.85 | 0.90 | 0.85 | +0.05 |
| `place_object_stand` | 0.85 | 0.95 | 0.90 | +0.05 |

These three positive MeanFlow-over-release10 cases are each only one episode
out of 20, so they should not be over-interpreted.

## Conclusion

The release10 follow-up changes the interpretation of the 2026-05-27 quick
eval. On the same 50-task clean20 set, `release_10` is much stronger than both
`release_1` and the joint-delta MeanFlow fine-tune:

```text
release_1  -> meanflow_1: +1.8 points
release_1  -> release_10: +20.1 points
meanflow_1 -> release_10: +18.3 points
```

Therefore this quick readout does not support the claim that the current
joint-delta MeanFlow conditioner has learned behavior close to the release
model's 10-step sampler. The safer conclusion is that it remains much closer
to the release one-step baseline, with a small positive average shift and
large task-specific variance.

## Follow-Up

Do not launch a full formal eval from this result alone. First inspect the
large-gap tasks above and verify whether failures are policy behavior,
simulator instability, or seed variance.

If continuing the direction, the next useful formal scope is:

```text
groups: release_1, release_10, meanflow_1
phases: clean, random
tasks: 50
episodes/task: 100
total online rollouts: 3 * 2 * 50 * 100 = 30000
```

That scope can distinguish whether the small MeanFlow lift survives more
episodes and randomized task configs.

## Runner Note

This run required pinning the quick-compare manager to physical GPU ids 6 and
7. `experiments/robotwin/run_robotwin_quick_compare.py` now accepts:

```text
GPU_IDS="6 7"
```

When `GPU_IDS` is unset, the runner preserves the previous behavior and uses
`range(NUM_GPUS)`.
