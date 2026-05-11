# 2026-05-11 FastWAM One-Day Quick One-Step Compare Plan

## 动机

完整 RoboTwin 对比需要 `4 steps * 50 tasks * 2 phases * 100 episodes = 40,000 episodes`，无法在一天内产出初步结论。第一性原理上，one-step diffusion / flow-matching 的第一轮问题不是覆盖全部任务，而是判断单步训练目标是否相对 release checkpoint 的 `1-step` sampler 有增益，并且是否接近 release checkpoint 的多步上界。

因此当天 quick plan 保留最小可解释对照，先拿方向性结果，再决定是否扩大到全任务全 episode。

## 核心对照

### Release sampler baseline

- `release-1`: release checkpoint, `num_inference_steps=1`
- `release-4`: release checkpoint, `num_inference_steps=4`
- `release-10`: release checkpoint, `num_inference_steps=10`

`release-2` 暂不进入 quick plan。它对完整退化曲线有价值，但对判断 one-step 目标是否有效不是必要点。

### One-step training targets

- `endpoint-1`: 当前已实现的 action-only one-step endpoint fine-tune, `num_inference_steps=1`
- `shortcut-1`: planned shortcut one-step fine-tune, `num_inference_steps=1`
- `meanflow-1`: planned mean-flow one-step fine-tune, `num_inference_steps=1`

当前代码状态：

- endpoint/action direct fine-tune 已实现并已有 quick checkpoint。
- shortcut 已实现并已有 10-step smoke checkpoint，可进入同 task set quick eval。
- mean-flow 未实现。

当天已先跑 `release-1/4/10 + endpoint-1`；后续把 `shortcut-1/meanflow-1` 作为方法实现后的同配置补跑项。`shortcut-1` 当前已具备补跑条件。

## Quick Task Set

先使用 8 个代表任务：

```text
click_bell
click_alarmclock
adjust_bottle
grab_roller
beat_block_hammer
dump_bin_bigbin
blocks_ranking_size
stack_blocks_two
```

覆盖意图：

- 快速接触/点击类：`click_bell`, `click_alarmclock`
- 简单抓取/移动：`adjust_bottle`, `grab_roller`
- 中等操作：`beat_block_hammer`, `dump_bin_bigbin`
- 排序/堆叠：`blocks_ranking_size`, `stack_blocks_two`

## Evaluation Scale

当天优先级：

1. `demo_clean`, `20 episodes/task`
2. 如 clean 结果趋势明确且仍有时间，再补 `demo_randomized`, `10 episodes/task`

第一轮工作量：

```text
4 groups * 8 tasks * 20 episodes = 640 episodes
```

对比 full baseline：

```text
40,000 episodes
```

约缩小 62.5 倍。

## Result Criteria

每个 task 和 overall 记录：

- success rate
- checkpoint tag
- `num_inference_steps`
- phase
- episode count

初步判断：

- 若 `endpoint-1 > release-1` 且接近 `release-4`，endpoint/action direct 目标值得扩大训练。
- 若 `endpoint-1` 接近或低于 `release-1`，优先实现 shortcut / mean-flow 或 teacher-student distillation。
- shortcut 和 mean-flow 后续必须在相同 quick task set 上补跑，才可横向比较。

## Execution Notes

当前 full baseline 会话：

```text
fastwam_baseline_20260511
```

切换到 quick plan 时应先停止该会话，保留 partial outputs：

```text
evaluate_results/robotwin/robotwin_uncond_3cam_384/20260511_robotwin_release_baseline_steps_1/
```

Quick run 输出建议：

```text
evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20/
```

## Follow-up Full Plan

quick plan 只产出方向性证据。完整计划仍应包含：

- release checkpoint: `steps=1,2,4,10`
- endpoint/action direct checkpoint: `steps=1`
- shortcut checkpoint: `steps=1`
- mean-flow checkpoint: `steps=1`
- 50 RoboTwin tasks
- clean + randomized
- 100 episodes/task
