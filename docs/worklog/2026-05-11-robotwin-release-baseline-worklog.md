# 2026-05-11 RoboTwin Release Baseline Worklog

## 动机

one-step diffusion / one-step flow-matching 实验需要先建立无训练 baseline。第一性原理上，后续 fine-tuned checkpoint 是否有效，只能通过同一 release checkpoint、同一 dataset stats、同一 RoboTwin 任务集合下的 `num_inference_steps` 退化曲线来判断。

本次 baseline 固定：

- checkpoint: `checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`
- dataset stats: `checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json`
- task config list: `third_party/RoboTwin/task_config/_eval_step_limit.yml`
- inference steps: `1 2 4 10`
- RoboTwin manager: `experiments/robotwin/run_robotwin_manager.py`

## 环境修复

RoboTwin smoke 初次失败在 cuRobo planner 初始化：

```text
AttributeError: module 'warp' has no attribute 'torch'
AttributeError: 'Robot' object has no attribute 'left_planner'
```

根因是 `curobo==0.7.6` 的代码路径调用 `wp.torch.device_from_torch(...)`，但本地 `warp-lang==1.13.0` 不再暴露 `warp.torch`。`left_planner` 是 planner 初始化失败后的连锁错误，不是根因。

已执行修复：

```bash
/DATA/disk2/wangchen/projects/FastWAM/.conda/fastwam/bin/python -m pip install 'warp-lang==0.10.1'
```

验证：

```text
has_warp_torch True
envs+curobo ok True 2.7.1+cu128 .../third_party/RoboTwin/envs/curobo/src/curobo/__init__.py
```

## Smoke 验证

单任务 smoke：

```bash
env PATH=/DATA/disk2/wangchen/projects/FastWAM/.conda/fastwam/bin:$PATH \
  DIFFSYNTH_DOWNLOAD_SOURCE=modelscope \
  DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk2/wangchen/projects/FastWAM/checkpoints \
  TOKENIZERS_PARALLELISM=false \
  MPLCONFIGDIR=/DATA/disk3/tmp/matplotlib-fastwam \
  python experiments/robotwin/eval_robotwin_single.py \
    task=robotwin_uncond_3cam_384_1e-4 \
    ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
    gpu_id=3 \
    EVALUATION.task_name=click_alarmclock \
    EVALUATION.task_config=demo_clean \
    EVALUATION.eval_num_episodes=1 \
    EVALUATION.num_inference_steps=1 \
    EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
    EVALUATION.output_dir=./evaluate_results/robotwin_baseline_smoke/steps_1
```

结果：

```text
click_alarmclock | fastwam_policy | demo_clean
Success rate: 1/1 => 100.0%
```

输出：

- `evaluate_results/robotwin/robotwin_uncond_3cam_384/steps_1/click_alarmclock/_result_clean.txt`
- `evaluate_results/robotwin/robotwin_uncond_3cam_384/steps_1/eval_click_alarmclock_20260511_174127.log`

## 完整 Baseline 启动

完整 release baseline 已在 tmux 会话启动：

```text
tmux session: fastwam_baseline_20260511
tmux log: /DATA/disk3/tmp/fastwam_baseline_20260511_tmux.log
```

启动命令：

```bash
env PATH=/DATA/disk2/wangchen/projects/FastWAM/.conda/fastwam/bin:$PATH \
  DIFFSYNTH_DOWNLOAD_SOURCE=modelscope \
  DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk2/wangchen/projects/FastWAM/checkpoints \
  TOKENIZERS_PARALLELISM=false \
  MPLCONFIGDIR=/DATA/disk3/tmp/matplotlib-fastwam \
  CKPT=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  STATS=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  NUM_GPUS=8 \
  MAX_TASKS_PER_GPU=1 \
  STEPS="1 2 4 10" \
  RUN_ID=20260511_robotwin_release_baseline \
  bash scripts/run_one_step_diffusion_baseline.sh robotwin EVALUATION.eval_num_episodes=100
```

当前已进入 `steps=1`：

```text
manager start tasks=50 gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7] max_tasks_per_gpu=1
```

首批任务：

- `adjust_bottle`
- `beat_block_hammer`
- `blocks_ranking_rgb`
- `blocks_ranking_size`
- `click_alarmclock`
- `click_bell`
- `dump_bin_bigbin`
- `grab_roller`

结果路径按 step 归档：

- `evaluate_results/robotwin/robotwin_uncond_3cam_384/20260511_robotwin_release_baseline_steps_1/`
- 后续预期：`..._steps_2/`, `..._steps_4/`, `..._steps_10/`

每个 step 完成后应检查：

- `summary.csv`
- `summary.json`
- `failed_tasks.txt`
- per-task `eval_*.log`

## 后续

1. 监控 `fastwam_baseline_20260511`，确认 `steps=1` 是否完成 clean/random 全任务。
2. 若某个 worker 失败，先看对应 `eval_<task>_*.log`，不要直接比较不完整 summary。
3. 完成后将 `1/2/4/10` 的 overall 和 per-task success rate 汇总到新的 QA 或 worklog。
