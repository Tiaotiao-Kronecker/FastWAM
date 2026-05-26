# 2026-05-12 LIBERO One-Step Pilot Worklog

## Baseline Context

Release step curve on the official LIBERO checkpoint:

| Method | Inference steps | Overall | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---:|---:|---:|---:|---:|---:|
| release_10 | 10 | 96.80 | 97.80 | 99.40 | 95.40 | 94.60 |
| release_4 | 4 | 96.90 | 97.60 | 99.80 | 96.40 | 93.80 |
| release_2 | 2 | 96.75 | 97.20 | 99.20 | 96.60 | 94.00 |
| release_1 | 1 | 95.80 | 96.60 | 99.20 | 95.20 | 92.20 |

This means the raw one-step release baseline is already strong. The pilot goal is therefore to check whether one-step fine-tuning improves over `release_1`, especially on `libero_10`.

## Added LIBERO One-Step Configs

- `configs/task/libero_one_step_shortcut_2cam224_1e-4.yaml`
- `configs/task/libero_one_step_meanflow_2cam224_1e-4.yaml`

Both initialize from:

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
```

Pilot settings:

```text
max_steps: 2000
batch_size: 1
num_workers: 0
learning_rate: 1e-4
save_every: 1000
save_training_state: false
```

`save_training_state=false` avoids writing full optimizer/accelerate state for pilot runs. The final `.pt` weight checkpoint is still saved and is sufficient for LIBERO eval.

## Text Embedding Cache

LIBERO training needs cached Wan/T5 text embeddings because training uses `model.load_text_encoder=false`.

Generated:

```text
data/text_embeds_cache/libero/*.t5_len128.wan22ti2v5b.pt
```

Count:

```text
40 prompts
```

## Smoke Checks

Training smoke:

| Method | Output checkpoint | Status |
|---|---|---|
| shortcut | `runs/libero_one_step_shortcut_smoke/checkpoints/weights/step_000001.pt` | passed |
| meanflow | `runs/libero_one_step_meanflow_smoke/checkpoints/weights/step_000001.pt` | passed |

Eval load-smoke:

| Method | Suite/task | Trials | Result |
|---|---|---:|---:|
| shortcut | libero_spatial task 0 | 1 | 1/1 |
| meanflow | libero_spatial task 0 | 1 | 1/1 |

The one-step wrapper checkpoints can be loaded by `experiments/libero/eval_libero_single.py`.

## Running Pilot Training

Started tmux sessions:

```text
libero_shortcut_pilot2k
libero_meanflow_pilot2k
```

Training logs:

```text
/DATA/disk3/tmp/fastwam_libero_one_step_logs/shortcut_pilot2k_20260512.log
/DATA/disk3/tmp/fastwam_libero_one_step_logs/meanflow_pilot2k_20260512.log
```

Training outputs:

```text
runs/libero_one_step_shortcut_pilot2k_20260512/checkpoints/weights/step_002000.pt
runs/libero_one_step_meanflow_pilot2k_20260512/checkpoints/weights/step_002000.pt
```

At around 20:10:

```text
shortcut: step 320/2000, speed about 0.53 step/s, ETA about 53 min
meanflow: step 560/2000, speed about 0.92 step/s, ETA about 26 min
```

## Automatic Eval Watcher

Started tmux session:

```text
libero_one_step_pilot_eval_chain
```

Watcher script:

```text
experiments/libero/run_one_step_pilot_eval_chain.sh
```

Watcher log:

```text
/DATA/disk3/tmp/fastwam_libero_one_step_logs/one_step_pilot_eval_chain_20260512.log
```

After both `step_002000.pt` checkpoints are complete, it will run full LIBERO eval sequentially:

```text
evaluate_results/libero/shortcut_1_pilot2k_20260512
evaluate_results/libero/meanflow_1_pilot2k_20260512
```

Eval settings:

```text
suites: libero_10, libero_goal, libero_spatial, libero_object
tasks: 40 total
trials: 50 per task
num_inference_steps: 1
save_rollout_video: false
GPUs: 0,2,3
workers_per_gpu: 2
```

## Final Results

Completed on 2026-05-12.

| Method | Overall | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---:|---:|---:|---:|---:|
| release_10 | 96.80 | 97.80 | 99.40 | 95.40 | 94.60 |
| release_1 | 95.80 | 96.60 | 99.20 | 95.20 | 92.20 |
| shortcut_1_2k | 95.00 | 96.20 | 98.40 | 93.60 | 91.80 |
| meanflow_1_2k | 6.70 | 10.60 | 3.20 | 12.40 | 0.60 |

Interpretation:

- `shortcut_1_2k` does not beat the raw `release_1` baseline. It is lower by 0.80 overall points and lower on every suite.
- `shortcut_1_2k` has mixed per-task movement: it improves some tasks, but loses more on others. The strongest improvements are `libero_goal/task6` (+12 points) and `libero_10/task0` (+10 points). The largest drops are `libero_10/task4` (-16 points) and `libero_goal/task9` (-16 points).
- `meanflow_1_2k` collapses. This is not a small tuning gap; it points to a likely objective/implementation mismatch for the current MeanFlow formulation.
- The raw one-step release checkpoint remains the strongest one-step baseline among the tested one-step variants.
