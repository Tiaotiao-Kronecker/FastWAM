# LIBERO Task Sets

These files are comma-separated task lists consumed by
`experiments/libero/run_libero_parallel_test.sh` through
`experiments/libero/run_libero_manager.py`.

Task-list files intentionally contain only `suite,task_id` rows. The shell
runner does not parse comments in task files.

## Sets

- `libero_long_official.txt`: all 10 tasks from the official `libero_10`
  suite. This is the least biased long-horizon slice of the standard LIBERO
  benchmark.
- `libero_long_horizon_v1.txt`: all `libero_10` tasks plus three additional
  standard LIBERO tasks where the existing release checkpoint showed a clear
  `10-step > 1-step` margin. This is intended for iteration, not paper-style
  reporting.
- `libero_gap_probe_v1.txt`: the seven tasks with the largest positive
  `release_10 - release_1` gap in the 2026-05-12 baseline run. Use it as a
  quick diagnostic probe only.

## Baseline Margins From Existing Runs

The numbers below come from:

- `evaluate_results/libero/release_baseline_20260512_steps1/summary.json`
- `evaluate_results/libero/release_baseline_20260512_steps10_v2/summary.json`

| Task set | Tasks | release_1 | release_10 | Gap |
| --- | ---: | ---: | ---: | ---: |
| `libero_long_official` | 10 | 92.20 | 94.60 | +2.40 |
| `libero_long_horizon_v1` | 13 | 91.85 | 95.54 | +3.69 |
| `libero_gap_probe_v1` | 7 | 89.71 | 96.57 | +6.86 |

Use `libero_long_official` for the clean long-suite readout. Use
`libero_long_horizon_v1` when the full LIBERO average is too saturated to show
one-step degradation. Keep `libero_gap_probe_v1` separate because it was
selected using existing baseline gaps and is therefore biased.
