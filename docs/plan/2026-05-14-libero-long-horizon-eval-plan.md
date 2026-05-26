# 2026-05-14 LIBERO Long-Horizon Eval Plan

## Motivation

The full LIBERO average is too saturated for one-step method development:

- `release_1`: 95.80
- `release_10`: 96.80
- gap: +1.00 point

This makes it hard to tell whether a one-step objective is improving motion
integration or only moving noise inside an already saturated benchmark.

## Eval Sets

The new task files live in `experiments/libero/task_sets/`.

| Set | Purpose | Tasks | release_1 | release_10 | Gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `libero_long_official` | clean official long-suite readout | 10 | 92.20 | 94.60 | +2.40 |
| `libero_long_horizon_v1` | iteration set with a larger one-step margin | 13 | 91.85 | 95.54 | +3.69 |
| `libero_gap_probe_v1` | quick diagnostic, intentionally gap-selected | 7 | 89.71 | 96.57 | +6.86 |

`libero_long_official` is unbiased because it is exactly the official
`libero_10` suite. `libero_long_horizon_v1` and `libero_gap_probe_v1` are
development probes, not paper-style benchmark replacements.

## Usage

Run the default long-horizon iteration set with the release checkpoint at
one-step inference:

```bash
NUM_INFERENCE_STEPS=1 \
TASK_SET=libero_long_horizon_v1 \
experiments/libero/run_libero_task_set_eval.sh
```

Run the same set at 10-step inference:

```bash
NUM_INFERENCE_STEPS=10 \
TASK_SET=libero_long_horizon_v1 \
RUN_TAG=libero_long_horizon_v1_release10 \
experiments/libero/run_libero_task_set_eval.sh
```

Run the clean official long suite:

```bash
NUM_INFERENCE_STEPS=1 \
TASK_SET=libero_long_official \
experiments/libero/run_libero_task_set_eval.sh
```

For a custom file:

```bash
TASK_FILE=experiments/libero/task_sets/libero_gap_probe_v1.txt \
NUM_INFERENCE_STEPS=1 \
experiments/libero/run_libero_task_set_eval.sh
```

## Notes

- The task files contain only `suite,task_id` lines because the shell runner
  does not parse comments.
- Repeating the same task in one task file is not supported because result
  filenames are keyed by `suite` and `task_id`.
- Increasing `NUM_TRIALS` above 50 repeats LIBERO init states in the current
  evaluator, so it increases compute but does not create genuinely new
  episodes.
- The manager now respects `MULTIRUN.task_file`; without that fix it always
  regenerated the full suite list.
