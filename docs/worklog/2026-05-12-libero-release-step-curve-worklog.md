# 2026-05-12 LIBERO Release Step Curve Worklog

## Baseline anchor: release_10

Finished evaluating the official Fast-WAM LIBERO release checkpoint:

- checkpoint: `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- dataset stats: `checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`
- output: `evaluate_results/libero/release_baseline_20260512_steps10_v2`
- suites: `libero_10`, `libero_goal`, `libero_spatial`, `libero_object`
- tasks: 40 total
- trials: 50 per task, 2000 total
- inference steps: 10
- rollout MP4 saving disabled after the mid-run multiworker switch
- failures: none

Result:

| Suite | Success Rate |
| --- | ---: |
| `libero_spatial` | 97.80 |
| `libero_object` | 99.40 |
| `libero_goal` | 95.40 |
| `libero_10` | 94.60 |
| Overall | 96.80 |

Comparison against the paper Table 2 Fast-WAM / Ours row:

| Suite | Paper Fast-WAM | Our release_10 | Delta |
| --- | ---: | ---: | ---: |
| Spatial | 98.2 | 97.8 | -0.4 |
| Object | 100.0 | 99.4 | -0.6 |
| Goal | 97.0 | 95.4 | -1.6 |
| Long / `libero_10` | 95.2 | 94.6 | -0.6 |
| Average | 97.6 | 96.8 | -0.8 |

Interpretation: the released-checkpoint reproduction is close enough to use as the LIBERO baseline anchor. The remaining gap is small and plausibly explained by seed/environment/version differences rather than a configuration-level mismatch.

## Step Curve Plan

Run the same release checkpoint with fewer inference steps:

1. `release_4`
2. `release_2`
3. `release_1`

Keep all other evaluation settings fixed: suites, trials, dataset stats, task instructions, text CFG scale, action horizon, replan steps, and LIBERO environment. This curve is the baseline for judging whether `shortcut_1` or `meanflow_1` improves over the untrained one-step sampler.
