# 2026-05-11 FastWAM Quick One-Step Results QA

## Q: 一天内 quick compare 结束了吗？

结束了。

证据：

```text
[2026-05-11 20:11:06] quick compare finished
```

输出位置：

```text
evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20/summary.csv
evaluate_results/robotwin_quick_one_step_compare/20260511_quick_clean20/summary.json
```

`failed_jobs.txt` 为 0 字节，说明 32 个 planned jobs 全部完成。

## Q: 当前 quick compare 的核心结论是什么？

第一性原理上，这轮实验只回答一个窄问题：当前已经实现的 one-step endpoint/action direct checkpoint，是否能在相同任务集合上超过 release checkpoint 的 1-step sampler。

结果是否定的：

| group | mean success |
| --- | ---: |
| release_1 | 0.9250 |
| release_4 | 0.9875 |
| release_10 | 0.9750 |
| endpoint_1 | 0.1937 |

`endpoint_1` 明显低于 `release_1`。

## Q: 这是否说明 one-step diffusion / mean-flow 方向无效？

不能这样解释。

这轮只测试了一个很弱的初步 checkpoint：`runs/robotwin_one_step_action_10step/checkpoints/weights/step_000010.pt`。它只训练了 10 step，并且是 action-only endpoint fine-tune。它失败说明这个具体目标和训练强度不足，不能推出 shortcut 或 mean-flow 也会失败。

## Q: 为什么 release_1 会这么强？

release checkpoint 的 `num_inference_steps=1` 并不是重新训练出来的 one-step policy，而是在原始强模型上减少采样步数。clean quick subset 的 8 个任务相对容易，所以 release 模型单步采样已经能达到 0.9250。

这也提高了后续 one-step 方法的门槛：新的 `shortcut_1` 或 `meanflow_1` 不只要能跑通，还要超过这个很强的 release 1-step baseline。

## Q: 下一步应该做什么？

优先实现并训练计划中的两个 one-step 目标：

- `shortcut_1`
- `meanflow_1`

然后用同一套 quick task set 复跑，这样结果才可横向比较。若 endpoint/action direct 还要保留，应先把训练步数显著增加，再判断是否值得扩大到 full benchmark。

## Q: 为什么 endpoint fine-tune 里的 velocity loss 和 endpoint loss 在 `sigma=1` 时等价？

当前 action flow 的定义是：

```text
x_sigma = (1 - sigma) * action + sigma * noise
target_velocity = noise - action
pred_endpoint = x_sigma - sigma * pred_velocity
```

把 `pred_endpoint` 减去真实 action：

```text
pred_endpoint - action
= (1 - sigma) * action + sigma * noise - sigma * pred_velocity - action
= sigma * (noise - action - pred_velocity)
= sigma * (target_velocity - pred_velocity)
```

因此：

```text
MSE(pred_endpoint, action)
= sigma^2 * MSE(pred_velocity, target_velocity)
```

当 `sigma=1` 时，两项完全是同一个误差：

```text
MSE(pred_endpoint, action)
= MSE(pred_velocity, noise - action)
```

所以当前 endpoint 训练里的

```text
0.5 * MSE(pred_velocity, noise - action)
+ 0.5 * MSE(pred_endpoint, action)
```

在固定 `sigma=1` 时不会提供两个独立监督信号，本质上就是同一个 velocity 约束。保留 endpoint loss 的价值主要是让日志直接暴露 action endpoint 误差，并为后续混合 timestep 或改成独立 endpoint head 留接口。
