# A1 Checkpoint Sweep Strict Gap Probe

Date: 2026-06-12

Run stamp: `20260612_144300`

Protocol:

- Task file: `experiments/libero/task_sets/libero_gap_probe_v1.txt`
- 7 tasks x 50 trials = 350 trials/checkpoint
- `seed=42`
- `trial_indices=null`
- `num_inference_steps=1`
- `policy_subprocess=false`
- `save_action_trace=true`
- `save_rollout_video=false`
- `MULTIRUN.num_gpus=1`
- `MULTIRUN.max_tasks_per_gpu=1`
- GPU7

All five checkpoints completed with `failed_tasks=0`.

## Overall Results

| Rank | Checkpoint | Success | Overall | LIBERO spatial | LIBERO goal | LIBERO-10 subset | vs 70k |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A1 step070000 | 333/350 | 95.14% | 49/50 | 98/100 | 186/200 | baseline |
| 2 | A1 step060000 | 327/350 | 93.43% | 49/50 | 97/100 | 181/200 | -6 |
| 2 | A1 step065000 | 327/350 | 93.43% | 50/50 | 97/100 | 180/200 | -6 |
| 4 | A1 step055000 | 323/350 | 92.29% | 47/50 | 99/100 | 177/200 | -10 |
| 5 | A1 step050000 | 319/350 | 91.14% | 48/50 | 92/100 | 179/200 | -14 |

## Task Breakdown

Successes out of 50 trials:

| Checkpoint | spatial 3 | goal 5 | goal 6 | 10-0 | 10-2 | 10-3 | 10-8 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 step050000 | 48 | 48 | 44 | 43 | 50 | 50 | 36 | 319 |
| A1 step055000 | 47 | 49 | 50 | 42 | 49 | 50 | 36 | 323 |
| A1 step060000 | 49 | 48 | 49 | 44 | 49 | 49 | 39 | 327 |
| A1 step065000 | 50 | 48 | 49 | 44 | 49 | 49 | 38 | 327 |
| A1 step070000 | 49 | 48 | 50 | 45 | 49 | 50 | 42 | 333 |

## Readout

Under the same strict 350-trial gap-probe, A1 70k is the best checkpoint among the tested 50k/55k/60k/65k/70k weights.

The 70k repeat exactly matches the previous strict 70k result from `20260611_190935`: `333/350 = 95.14%`. This makes the 70k baseline stable under this eval wrapper.

Earlier checkpoints do improve monotonically from 50k to 60k/65k, but neither 60k nor 65k reaches 70k. The remaining 70k advantage is mostly in the LIBERO-10 subset, especially `libero_10_8`:

- 50k: 36/50
- 55k: 36/50
- 60k: 39/50
- 65k: 38/50
- 70k: 42/50

Conclusion: the training metrics plateau around 50k-65k did not translate into a better closed-loop checkpoint on this probe. The degradation observed at 80k is therefore best interpreted as post-70k continuation damage, not evidence that 70k itself was already past the best point among the saved A1 checkpoints.
