# A1 step010000 gap-probe archive

Date archived: 2026-06-05

## Scope

This archive summarizes the three A1 `step_010000.pt` checkpoints copied from H200-1 and evaluated on H200-2.

Task set:

- `experiments/libero/task_sets/libero_gap_probe_v1.txt`
- 7 tasks
- 50 trials per task
- `EVALUATION.num_inference_steps=1`
- rollout video/action trace disabled

Final successful evals:

- `a1_step010000_control_gap_probe_serial_v1_steps1_50trials_20260604`
- `a1_step010000_residual_only_gap_probe_serial_v1_steps1_50trials_20260604`
- `a1_step010000_residual_clip025_gap_probe_v1_steps1_50trials_20260604`

All final `failed_tasks.txt` files are empty.

Important caveat:

- `control` and `residual_only` final serial runs used `policy_subprocess=false` because the serial wrapper did not export `EXTRA_MANAGER_ARGS` into the child launcher.
- `residual_clip025` used `policy_subprocess=true`, since it was the already-running eval launched before the serial queue.
- Earlier concurrent / diagnostic attempts for `control` and `residual_only` hit `rc=134` on `libero_goal,6`; the final serial runs completed successfully.

## Main Results

Same 7-task gap-probe scope:

| method | total | avg task SR | spatial | goal | libero_10 subset | vs A1 step070k |
|---|---:|---:|---:|---:|---:|---:|
| A1 step070k overlap | 333/350 | 95.14 | 98.00 | 98.00 | 93.00 | baseline |
| control step010k | 320/350 | 91.43 | 90.00 | 96.00 | 89.50 | -13 succ / -3.71 pts |
| residual_only step010k | 320/350 | 91.43 | 98.00 | 98.00 | 86.50 | -13 succ / -3.71 pts |
| residual_clip025 step010k | 317/350 | 90.57 | 94.00 | 98.00 | 86.00 | -16 succ / -4.57 pts |

Interpretation:

- None of the three `step010k` checkpoints beats A1 step070k on the same seven gap-probe tasks.
- `control` ties `residual_only` on total successes, and has the best `libero_10` subset score among the three `step010k` variants.
- `residual_only` matches A1 step070k on `spatial` and `goal`, but loses on `libero_10`.
- `residual_clip025` is the lowest of the three on this gap-probe run.

## Task-Level Breakdown

Success rate in percent:

| task | A1 step070k | control | residual_only | residual_clip025 |
|---|---:|---:|---:|---:|
| `libero_goal_6` | 100.00 | 94.00 | 96.00 | 98.00 |
| `libero_10_0` | 90.00 | 86.00 | 90.00 | 82.00 |
| `libero_goal_5` | 96.00 | 98.00 | 100.00 | 98.00 |
| `libero_10_2` | 98.00 | 94.00 | 90.00 | 88.00 |
| `libero_spatial_3` | 98.00 | 90.00 | 98.00 | 94.00 |
| `libero_10_8` | 84.00 | 78.00 | 66.00 | 80.00 |
| `libero_10_3` | 100.00 | 100.00 | 100.00 | 94.00 |

## Release Baselines

Release baselines are recorded in:

- `docs/worklog/2026-05-12-libero-one-step-pilot-worklog.md`

Those are full official LIBERO 4-suite runs, not the same seven-task gap-probe subset, so they are directional rather than strict apples-to-apples comparisons.

| method | overall | spatial | object | goal | libero_10 |
|---|---:|---:|---:|---:|---:|
| release_10 | 96.80 | 97.80 | 99.40 | 95.40 | 94.60 |
| release_1 | 95.80 | 96.60 | 99.20 | 95.20 | 92.20 |

Directional comparison:

- The `step010k` gap-probe averages (`90.57-91.43`) are below the release full-suite overall baselines.
- The largest gap is still in `libero_10`, where the `step010k` variants score `86.00-89.50` on the selected subset.

## Source Artifacts

Current `step010k` summaries:

- `evaluate_results/libero/a1_step010000_control_gap_probe_serial_v1_steps1_50trials_20260604/summary.json`
- `evaluate_results/libero/a1_step010000_residual_only_gap_probe_serial_v1_steps1_50trials_20260604/summary.json`
- `evaluate_results/libero/a1_step010000_residual_clip025_gap_probe_v1_steps1_50trials_20260604/summary.json`

A1 step070k comparison source:

- `evaluate_results/libero/a1_lora_eqanchor_step070000_long_horizon_policy_subprocess_single_gpu6_20260603_094011/summary.json`

Machine-readable archived tables:

- `gap_probe_7task_comparison.csv`
- `gap_probe_task_breakdown.csv`
