# 2026-05-11 FastWAM One-Step Shortcut Worklog

## 动机

`endpoint_1` quick compare 明显低于 release 1-step baseline。第一性原理上，endpoint 训练只监督“从纯噪声直接到 action 端点”，没有显式学习“大步长一步”和“多个小步”之间的一致性。因此下一步先实现 action-only `shortcut_1`，在不改 video branch 的前提下加入 shortcut step-size conditioning 和长跳自一致 loss。

## Loss 关系澄清

当前 action flow:

```text
x_sigma = (1 - sigma) * action + sigma * noise
target_velocity = noise - action
pred_endpoint = x_sigma - sigma * pred_velocity
```

因此：

```text
pred_endpoint - action = sigma * (target_velocity - pred_velocity)
```

当 `sigma=1` 时，endpoint loss 与 velocity loss 完全等价。当前 endpoint/action direct 训练里的两项 loss 不是两个独立监督信号。

## 实现内容

新增模型：

```text
src/fastwam/models/wan22/fastwam_one_step_shortcut.py
```

新增 runtime factory：

```text
fastwam.runtime.create_fastwam_one_step_shortcut
```

新增配置：

```text
configs/model/fastwam_one_step_shortcut.yaml
configs/task/robotwin_one_step_shortcut_3cam_384_1e-4.yaml
```

更新 quick compare runner：

```text
experiments/robotwin/run_robotwin_quick_compare.py
```

新增 group：

```text
shortcut_1
```

默认 checkpoint：

```text
runs/robotwin_one_step_shortcut_10step/checkpoints/weights/step_000010.pt
```

## Shortcut Objective

当前第一版仍为 action-only：

- video expert 冻结。
- 从 release checkpoint 初始化。
- 起点 `sigma=1`。
- shortcut step size `d=1`。
- action expert 额外接收 `d` 的 sinusoidal embedding，并把它加到 action time modulation。

训练 loss:

```text
loss_action_velocity
loss_action_endpoint
loss_shortcut_consistency
loss_shortcut_half_velocity
```

其中 shortcut consistency 使用：

```text
large jump velocity ~= average(two half-step velocities)
```

这比 endpoint direct 多了一个约束：一步长跳的预测要和两次半步预测一致。

## Verification

静态检查：

```bash
.conda/fastwam/bin/python -m py_compile \
  src/fastwam/models/wan22/fastwam_one_step_shortcut.py \
  src/fastwam/runtime.py \
  experiments/robotwin/run_robotwin_quick_compare.py
```

Hydra 配置解析：

```bash
.conda/fastwam/bin/python scripts/train.py task=robotwin_one_step_shortcut_3cam_384_1e-4 --cfg job
```

1-step smoke:

```text
output: runs/robotwin_one_step_shortcut_smoke/checkpoints/weights/step_000001.pt
loss: 0.0023
loss_action_endpoint: 0.0008
loss_action_velocity: 0.0008
loss_shortcut_consistency: 0.0002
loss_shortcut_half_velocity: 0.0006
```

10-step check:

```text
output: runs/robotwin_one_step_shortcut_10step/checkpoints/weights/step_000010.pt
final loss: 0.1021
loss_action_endpoint: 0.0332
loss_action_velocity: 0.0332
loss_shortcut_consistency: 0.0157
loss_shortcut_half_velocity: 0.0201
speed: 0.39 step/s
```

RoboTwin eval smoke:

```text
task: click_bell
phase: demo_clean
episodes: 2
success: 2/2
result: evaluate_results/robotwin/robotwin_one_step_shortcut_10step_checkpoints/20260511_shortcut_eval_smoke/click_bell/_result_clean.txt
```

## Next

用同一套 quick task set 跑 `shortcut_1`：

```text
GROUPS=shortcut_1
TASKS="click_bell click_alarmclock adjust_bottle grab_roller beat_block_hammer dump_bin_bigbin blocks_ranking_size stack_blocks_two"
PHASES=clean
EPISODES=20
```

`meanflow_1` 仍待实现。它应作为下一种 one-step objective，在同一 quick task set 上横向比较。
