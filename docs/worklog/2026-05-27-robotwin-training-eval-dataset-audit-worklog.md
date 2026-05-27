# 2026-05-27 RoboTwin Training/Eval Dataset Audit Worklog

## Purpose

Archive the RoboTwin dataset accounting used for the current FastWAM MeanFlow
joint-delta conditioner evaluation. The main clarification is that "dataset"
can mean different things in this repo:

- one LeRobot dataset root used by training
- 50 RoboTwin simulator tasks used by eval
- 27,500 demonstration episodes recorded in dataset stats
- 20 or 100 online rollout episodes per eval task
- 2,000 optimizer steps in the MeanFlow fine-tune

These should not be compared as if they were the same unit.

## Training Dataset

The RoboTwin training config is:

```text
configs/data/robotwin.yaml
```

Both train and val point to one LeRobot dataset root:

```yaml
dataset_dirs:
  - ./data/robotwin2.0/robotwin2.0
```

So at the config level this is one dataset root, not 50 independent dataset
directories. The root contains RoboTwin task demonstrations.

The model-side data shape is:

```text
camera streams: cam_high, cam_left_wrist, cam_right_wrist
action dim: 14
state/proprio dim: 14
num_frames: 33
action_video_freq_ratio: 4
video frames per sample: 9
action rows per sample: 32
final video layout: concat_multi_camera=robotwin, video_size=[384, 320]
```

The normalization stats bundled with both release and MeanFlow run record:

```text
num_episodes: 27,500
num_transition: 6,075,103
```

The training log for the RoboTwin MeanFlow joint-delta run records:

```text
Train/val dataset size: 6,011,575 / 63,528
```

With `val_set_proportion=0.01`, the episode split is approximately:

```text
train episodes: 27,225
val episodes: 275
```

Note: in the current workspace, `./data/robotwin2.0/robotwin2.0` is not mounted
under the repo checkout. The counts above are confirmed from run config,
`dataset_stats.json`, and `train.log`, not by scanning the dataset root live.

## Release Assets

Release policy:

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

Release dataset stats:

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

The release stats and the MeanFlow run stats agree on:

```text
num_episodes: 27,500
num_transition: 6,075,103
```

## MeanFlow Joint-Delta Fine-Tune

Run:

```text
runs/robotwin_one_step_meanflow_joint_conditioner_3cam_384_1e-4/joint_cond_fd_2k_20260525_prefix2200
```

Complete triplet for eval:

```text
config.yaml
dataset_stats.json
checkpoints/weights/step_002000.pt
```

Key config:

```yaml
resume: ./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
max_steps: 2000
batch_size: 1
num_workers: 0
model.one_step_meanflow.objective: finite_difference
model.one_step_meanflow.trainable_scope: conditioner
model.one_step_meanflow.conditioner_mode: joint_delta
model.one_step_meanflow.train_proprio_encoder: false
model.one_step_meanflow.freeze_video_expert: true
model.one_step_meanflow.derivative_epsilon: 0.05
```

This means the fine-tune used the full RoboTwin training pool as the sampling
source, but only executed:

```text
2000 optimizer steps * batch_size 1 * world_size 1 = 2000 sampled training windows
```

It did not train for a full pass over the 6,011,575 training samples.

The run suffix `prefix2200` is not a dataset count. It refers to precomputing
text embeddings for the deterministic sampler prefix used by the 2k-step run.
The relevant precompute log states:

```text
Loaded 2016 unique prompts from /DATA/disk3/tmp/robotwin_jointdelta_2200_train_prompts_20260525.txt
```

So:

```text
prefix2200 = sampled-prefix prompt-cache coverage
2016 = unique prompts in that prefix
2000 = optimizer steps
1 = LeRobot dataset root
27,500 = recorded demonstration episodes in stats
```

## Eval Design

The active compare runner is:

```text
experiments/robotwin/run_robotwin_quick_compare.py
```

It dispatches single-task jobs through:

```text
experiments/robotwin/eval_robotwin_single.py
```

Each single-task job calls RoboTwin's official entrypoint:

```text
third_party/RoboTwin/script/eval_policy.py
```

The current full task list is:

```text
third_party/RoboTwin/task_config/_eval_step_limit.yml
```

This file has 50 tasks. Its values are per-task eval step limits, not dataset
sizes.

The current compare run is:

```text
evaluate_results/robotwin_quick_one_step_compare/20260527_meanflow_jointdelta_50task_clean20
```

Current scope:

```text
groups: release_1, meanflow_1
phase: clean -> demo_clean
episodes/task: 20
tasks: 50
total jobs: 2 groups * 1 phase * 50 tasks = 100 jobs
total rollouts if complete: 2 * 50 * 20 = 2000 online eval episodes
```

The eval metric is task-level mean success:

```text
task_success_rate = successes / eval_num_episodes
group_mean = mean(task_success_rate over completed tasks)
```

This is not an episode-weighted mean across all rollout attempts, though with a
fixed 20 episodes/task the two means are numerically equivalent after every
task has completed.

The earlier 8-task quick eval was only a screening subset. It should not be
treated as the final RoboTwin conclusion, and `dump_bin_bigbin` alone is not a
valid basis for judging the MeanFlow fine-tune.

## Formal Eval Recommendation

The current 50-task clean20 run is the right next diagnostic because it covers
the full RoboTwin task set at modest episode count. It can answer whether
MeanFlow is broadly competitive with `release_1` on clean scenes.

For a stronger report, follow with:

```text
50 tasks
clean + randomized phases
100 episodes/task
release_1 vs meanflow_1
same dataset_stats.json
same num_inference_steps=1
```

That formal setting would be:

```text
2 groups * 2 phases * 50 tasks * 100 episodes = 20,000 online rollout episodes
```

Use the same task-level mean convention and report per-task deltas.

