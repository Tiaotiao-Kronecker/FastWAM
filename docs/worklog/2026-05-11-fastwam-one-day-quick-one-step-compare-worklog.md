# 2026-05-11 FastWAM One-Day Quick One-Step Compare Worklog

## 动机

用户要求在一天内拿到 one-step diffusion / flow-matching 的初步对比结果。第一性原理上，当天实验应回答“one-step 训练目标是否优于 release 1-step sampler，并接近 release 多步上界”，而不是完整覆盖所有任务、所有 phase 和所有 episode。

## 计划归档

已新增计划：

- `docs/plan/2026-05-11-fastwam-one-day-quick-one-step-compare-plan.md`

计划矩阵：

- `release-1`
- `release-4`
- `release-10`
- `endpoint-1`
- `shortcut-1` planned
- `meanflow-1` planned

当前代码状态：

- endpoint/action direct fine-tune 已实现并有 checkpoint。
- shortcut 尚未实现。
- mean-flow 尚未实现。

因此当天先执行 `release-1/4/10 + endpoint-1`，后续在 shortcut / mean-flow 实现后同配置补跑。

## Quick Scope

第一轮：

- phase: `demo_clean`
- episodes/task: 20
- tasks:
  - `click_bell`
  - `click_alarmclock`
  - `adjust_bottle`
  - `grab_roller`
  - `beat_block_hammer`
  - `dump_bin_bigbin`
  - `blocks_ranking_size`
  - `stack_blocks_two`

第一轮工作量：

```text
4 groups * 8 tasks * 20 episodes = 640 episodes
```

## Runtime Notes

完整 baseline 会话 `fastwam_baseline_20260511` 已不再符合当天 quick plan 的目标，应停止以释放 8 张 GPU。已完成的 partial outputs 保留在：

```text
evaluate_results/robotwin/robotwin_uncond_3cam_384/20260511_robotwin_release_baseline_steps_1/
```

后续 quick run 输出使用：

```text
evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20/
```

## Runner

新增 quick compare manager：

```text
experiments/robotwin/run_robotwin_quick_compare.py
```

默认配置：

- `GROUPS="release_1 release_4 release_10 endpoint_1"`
- `TASKS="click_bell click_alarmclock adjust_bottle grab_roller beat_block_hammer dump_bin_bigbin blocks_ranking_size stack_blocks_two"`
- `PHASES="clean"`
- `EPISODES=20`
- `NUM_GPUS=8`
- `MAX_TASKS_PER_GPU=1`

已验证：

```bash
.conda/fastwam/bin/python -m py_compile experiments/robotwin/run_robotwin_quick_compare.py
```

## Execution

已停止 full baseline tmux 会话：

```text
fastwam_baseline_20260511
```

已启动 quick compare：

```text
tmux session: fastwam_quick_compare_20260511
tmux log: /DATA/disk3/tmp/fastwam_quick_compare_20260511_tmux.log
manager log: evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20/manager.log
summary csv: evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20/summary.csv
summary json: evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20/summary.json
```

启动命令：

```bash
env PATH=/DATA/disk2/wangchen/projects/FastWAM/.conda/fastwam/bin:$PATH \
  DIFFSYNTH_DOWNLOAD_SOURCE=modelscope \
  DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk2/wangchen/projects/FastWAM/checkpoints \
  TOKENIZERS_PARALLELISM=false \
  MPLCONFIGDIR=/DATA/disk3/tmp/matplotlib-fastwam \
  RUN_ID=20260511_quick_clean20 \
  OUTPUT_ROOT=./evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20 \
  GROUPS="release_1 release_4 release_10 endpoint_1" \
  PHASES="clean" \
  EPISODES=20 \
  NUM_GPUS=8 \
  MAX_TASKS_PER_GPU=1 \
  python experiments/robotwin/run_robotwin_quick_compare.py
```

首批已启动：

```text
release_1, clean, 8 tasks, GPU 0-7
```

截至启动检查，worker 已进入 episode 阶段，`click_alarmclock` 已开始输出 success rate。
