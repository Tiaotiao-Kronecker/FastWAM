# Strict Gap-Probe Rerun, 6 Weights, GPU7

Date: 2026-06-11 to 2026-06-12
Host: H200-2
Run stamp: `20260611_190935`
Task file: `experiments/libero/task_sets/libero_gap_probe_v1.txt`
Protocol: 7 tasks x 50 trials = 350 trials per weight
Shared eval settings: `seed=42`, `trial_indices=null`, `policy_subprocess=false`, `save_action_trace=true`, `save_rollout_video=false`, `MULTIRUN.num_gpus=1`, `MULTIRUN.max_tasks_per_gpu=1`, GPU7

Note: `release1` and A1-family weights use `num_inference_steps=1`; `release10` uses `num_inference_steps=10` by definition.

## Overall Results

| Rank | Weight | Successes | Trials | Overall % | Spatial | Goal | LIBERO-10 | Delta vs A1 70k |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | release10 | 338 | 350 | 96.57 | 49/50 | 99/100 | 190/200 | +5 |
| 2 | A1 70k | 333 | 350 | 95.14 | 49/50 | 98/100 | 186/200 | 0 |
| 3 | A1.3 full clip0.25 70.3k | 329 | 350 | 94.00 | 49/50 | 97/100 | 183/200 | -4 |
| 4 | A1.2 full residual-only 70.3k | 326 | 350 | 93.14 | 50/50 | 97/100 | 179/200 | -7 |
| 5 | A1 full-control 80k | 324 | 350 | 92.57 | 48/50 | 97/100 | 179/200 | -9 |
| 6 | release1 | 314 | 350 | 89.71 | 47/50 | 89/100 | 178/200 | -19 |

## Task Breakdown

| Weight | spatial3 | goal5 | goal6 | l10_0 | l10_2 | l10_3 | l10_8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| release1 | 47 | 46 | 43 | 43 | 45 | 46 | 44 |
| release10 | 49 | 50 | 49 | 48 | 48 | 48 | 46 |
| A1 70k | 49 | 48 | 50 | 45 | 49 | 50 | 42 |
| A1 full-control 80k | 48 | 47 | 50 | 41 | 50 | 50 | 38 |
| A1.2 full residual-only 70.3k | 50 | 48 | 49 | 44 | 49 | 47 | 39 |
| A1.3 full clip0.25 70.3k | 49 | 48 | 49 | 42 | 50 | 49 | 42 |

Each task has 50 trials.

## Key Readout

The strict rerun confirms that A1 full-control 80k underperforms A1 70k on the same eval wrapper: 324/350 vs 333/350, a drop of 9 successes or 2.57 percentage points. This is not explained by mixing in the older weight-only continuation result.

The decline is concentrated in LIBERO-10 task0 and task8:

- A1 70k vs full-control 80k: l10_0 drops 45 -> 41, l10_8 drops 42 -> 38.
- Clip0.25 recovers task8 back to 42, but does not recover task0 enough, ending at 329/350.
- Residual-only improves task0 relative to control, but loses on task3, ending at 326/350.

Release10 is still the strongest on this probe at 338/350. A1 70k remains the best one-step/A1-family checkpoint among the six.

## Implication For Next Training

Do not treat the 70k -> 80k control continuation as a guaranteed improvement path. The full-state continuation itself appears to move performance down on this probe.

Among the tested A1 attribution branches, clip0.25 is the best of the three continuation variants, but it still does not beat the original A1 70k. The next directional experiment should therefore prioritize the planned interval-mixture branch (`A1.4-mix` and possibly `mix+clip`) from the A1 70k full state, evaluated under this same strict protocol before any 30k continuation or endpoint/rank/data-upweight branch.

Because this gap-probe set does not include LIBERO-10 task4/6/9, use it for first-pass attribution ranking, then confirm the winner against a broader or targeted hard-task set.
