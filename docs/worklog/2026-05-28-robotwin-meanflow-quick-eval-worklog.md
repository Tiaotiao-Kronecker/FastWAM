# 2026-05-28 RoboTwin MeanFlow Quick Eval Worklog

## Purpose

Archive the completed RoboTwin quick evaluation for the MeanFlow joint-delta
conditioner fine-tune before launching a larger formal eval.

This run is a broad quick diagnostic over the current 50-task RoboTwin eval
set. It is not the final formal result because it only covers the clean phase
and uses 20 online episodes per task.

## Run Metadata

Run:

```text
evaluate_results/robotwin_quick_one_step_compare/20260527_meanflow_jointdelta_50task_clean20
```

Timing from `manager.log`:

```text
start:  2026-05-27 09:51:07
finish: 2026-05-28 02:29:49
```

Runner:

```text
experiments/robotwin/run_robotwin_quick_compare.py
experiments/robotwin/eval_robotwin_single.py
third_party/RoboTwin/script/eval_policy.py
```

Scope:

```text
groups: release_1, meanflow_1
phase: clean -> demo_clean
tasks: 50
episodes/task: 20
num_inference_steps: 1
num_gpus: 2
max_tasks_per_gpu: 1
total jobs: 2 groups * 1 phase * 50 tasks = 100
total online rollouts: 2 groups * 50 tasks * 20 episodes = 2000
```

Completion checks:

```text
summary.csv: 101 lines = 100 result records + header
failed_jobs.txt: 0 lines
manager status: finished normally
```

Primary result files:

```text
evaluate_results/robotwin_quick_one_step_compare/20260527_meanflow_jointdelta_50task_clean20/summary.csv
evaluate_results/robotwin_quick_one_step_compare/20260527_meanflow_jointdelta_50task_clean20/summary.json
evaluate_results/robotwin_quick_one_step_compare/20260527_meanflow_jointdelta_50task_clean20/manager.log
evaluate_results/robotwin_quick_one_step_compare/20260527_meanflow_jointdelta_50task_clean20/failed_jobs.txt
```

## Compared Checkpoints

Release baseline:

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

MeanFlow joint-delta conditioner:

```text
runs/robotwin_one_step_meanflow_joint_conditioner_3cam_384_1e-4/joint_cond_fd_2k_20260525_prefix2200/checkpoints/weights/step_002000.pt
```

Dataset stats used by the quick compare:

```text
runs/robotwin_one_step_meanflow_joint_conditioner_3cam_384_1e-4/joint_cond_fd_2k_20260525_prefix2200/dataset_stats.json
```

The previous dataset audit records that these stats agree with the release
RoboTwin stats on the dataset counts and normalization source:

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
group_mean = mean(task_success_rate over 50 completed tasks)
```

Because every task has exactly 20 episodes in this run, this is numerically
equivalent to the episode-weighted success rate over 1000 rollouts per group.

Do not mix this run's `release_1` mean with release means from other eval
scopes. This record's release baseline is only the paired clean20 baseline for
`20260527_meanflow_jointdelta_50task_clean20`.

## Aggregate Result

| Group | Completed Tasks | Rollouts | Successes | Mean Success |
| --- | ---: | ---: | ---: | ---: |
| `release_1` | 50 / 50 | 1000 | 722 | 0.7220 |
| `meanflow_1` | 50 / 50 | 1000 | 740 | 0.7400 |
| delta | - | - | +18 | +0.0180 |

Task-level deltas:

```text
improved: 20
equal:    14
worse:    16
```

Quick interpretation:

- MeanFlow is slightly above the paired release baseline on this 50-task
  clean20 run: `+1.8` percentage points.
- The improvement is broad enough to justify a larger formal eval, but not
  strong enough to treat this as a final conclusion.
- Large regressions exist on specific tasks and should be inspected before
  over-claiming the result.

## Largest Positive Deltas

| Task | Release | MeanFlow | Delta |
| --- | ---: | ---: | ---: |
| `scan_object` | 0.25 | 0.85 | +0.60 |
| `put_object_cabinet` | 0.25 | 0.55 | +0.30 |
| `move_can_pot` | 0.40 | 0.70 | +0.30 |
| `place_fan` | 0.30 | 0.55 | +0.25 |
| `beat_block_hammer` | 0.70 | 0.95 | +0.25 |
| `place_shoe` | 0.80 | 1.00 | +0.20 |
| `place_phone_stand` | 0.80 | 1.00 | +0.20 |
| `place_bread_basket` | 0.60 | 0.75 | +0.15 |
| `turn_switch` | 0.45 | 0.60 | +0.15 |
| `place_bread_skillet` | 0.80 | 0.95 | +0.15 |

## Largest Negative Deltas

| Task | Release | MeanFlow | Delta |
| --- | ---: | ---: | ---: |
| `handover_mic` | 0.85 | 0.35 | -0.50 |
| `dump_bin_bigbin` | 0.95 | 0.70 | -0.25 |
| `pick_diverse_bottles` | 1.00 | 0.75 | -0.25 |
| `open_laptop` | 0.90 | 0.75 | -0.15 |
| `rotate_qrcode` | 0.75 | 0.60 | -0.15 |
| `hanging_mug` | 0.35 | 0.20 | -0.15 |
| `place_mouse_pad` | 0.80 | 0.70 | -0.10 |
| `stack_blocks_three` | 0.45 | 0.35 | -0.10 |
| `place_a2b_left` | 0.95 | 0.85 | -0.10 |
| `place_container_plate` | 0.95 | 0.85 | -0.10 |

## Full Per-Task Delta Table

Sorted by `meanflow_1 - release_1`.

| Task | Release | MeanFlow | Delta |
| --- | ---: | ---: | ---: |
| `handover_mic` | 0.85 | 0.35 | -0.50 |
| `dump_bin_bigbin` | 0.95 | 0.70 | -0.25 |
| `pick_diverse_bottles` | 1.00 | 0.75 | -0.25 |
| `open_laptop` | 0.90 | 0.75 | -0.15 |
| `rotate_qrcode` | 0.75 | 0.60 | -0.15 |
| `hanging_mug` | 0.35 | 0.20 | -0.15 |
| `place_mouse_pad` | 0.80 | 0.70 | -0.10 |
| `stack_blocks_three` | 0.45 | 0.35 | -0.10 |
| `place_a2b_left` | 0.95 | 0.85 | -0.10 |
| `place_container_plate` | 0.95 | 0.85 | -0.10 |
| `move_pillbottle_pad` | 0.85 | 0.75 | -0.10 |
| `stack_bowls_three` | 0.70 | 0.60 | -0.10 |
| `move_playingcard_away` | 1.00 | 0.95 | -0.05 |
| `stack_blocks_two` | 1.00 | 0.95 | -0.05 |
| `handover_block` | 0.10 | 0.05 | -0.05 |
| `place_object_basket` | 0.95 | 0.90 | -0.05 |
| `adjust_bottle` | 1.00 | 1.00 | +0.00 |
| `click_alarmclock` | 1.00 | 1.00 | +0.00 |
| `click_bell` | 1.00 | 1.00 | +0.00 |
| `grab_roller` | 1.00 | 1.00 | +0.00 |
| `move_stapler_pad` | 0.30 | 0.30 | +0.00 |
| `pick_dual_bottles` | 1.00 | 1.00 | +0.00 |
| `place_a2b_right` | 0.95 | 0.95 | +0.00 |
| `place_cans_plasticbox` | 0.45 | 0.45 | +0.00 |
| `place_empty_cup` | 1.00 | 1.00 | +0.00 |
| `place_object_scale` | 0.85 | 0.85 | +0.00 |
| `press_stapler` | 1.00 | 1.00 | +0.00 |
| `put_bottles_dustbin` | 0.10 | 0.10 | +0.00 |
| `shake_bottle` | 1.00 | 1.00 | +0.00 |
| `shake_bottle_horizontally` | 1.00 | 1.00 | +0.00 |
| `blocks_ranking_size` | 0.80 | 0.85 | +0.05 |
| `open_microwave` | 0.05 | 0.10 | +0.05 |
| `blocks_ranking_rgb` | 0.95 | 1.00 | +0.05 |
| `lift_pot` | 0.75 | 0.80 | +0.05 |
| `place_can_basket` | 0.50 | 0.55 | +0.05 |
| `place_burger_fries` | 0.95 | 1.00 | +0.05 |
| `stack_bowls_two` | 0.85 | 0.90 | +0.05 |
| `stamp_seal` | 0.50 | 0.55 | +0.05 |
| `place_object_stand` | 0.85 | 0.95 | +0.10 |
| `place_dual_shoes` | 0.35 | 0.45 | +0.10 |
| `place_bread_skillet` | 0.80 | 0.95 | +0.15 |
| `turn_switch` | 0.45 | 0.60 | +0.15 |
| `place_bread_basket` | 0.60 | 0.75 | +0.15 |
| `place_phone_stand` | 0.80 | 1.00 | +0.20 |
| `place_shoe` | 0.80 | 1.00 | +0.20 |
| `beat_block_hammer` | 0.70 | 0.95 | +0.25 |
| `place_fan` | 0.30 | 0.55 | +0.25 |
| `move_can_pot` | 0.40 | 0.70 | +0.30 |
| `put_object_cabinet` | 0.25 | 0.55 | +0.30 |
| `scan_object` | 0.25 | 0.85 | +0.60 |

## Formal Eval Follow-Up

Recommended next formal scope:

```text
groups: release_1, meanflow_1
phases: clean, random
tasks: 50
episodes/task: 100
total online rollouts: 2 * 2 * 50 * 100 = 20000
```

Run it only when GPU availability is stable. The latest GPU check before this
record showed the machine was not a good formal-eval window because GPUs 2-7
were at 100% utilization and GPUs 0-1 were also occupied by other Python
services or jobs.

Before reporting the quick eval externally, inspect at least:

```text
handover_mic
dump_bin_bigbin
pick_diverse_bottles
scan_object
put_object_cabinet
move_can_pot
```

Those tasks dominate the positive and negative tails and can change the
narrative if their failures come from simulator instability, seed variance, or
a task-specific policy behavior change.
