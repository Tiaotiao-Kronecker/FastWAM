# 2026-05-15 LIBERO Release Latency QA

## Q1: 之前说的 `1.55x` 提速，和这次正式测到的 `4.46x` / `9.77x` 是什么关系？

它们不是同一个口径。

| Metric | Ratio | What it measures |
| --- | ---: | --- |
| Full LIBERO rollout wall-clock | `1.5499x` | `release_10` 总评测时间 / `release_1` 总评测时间 |
| Pure `infer_action` end-to-end | `4.4595x` | 单次动作推理的端到端时间比 |
| Pure action denoise loop | `9.7692x` | 只看 action 去噪循环本身的时间比 |

## Q2: `1.55x` 为什么这么小？

因为 full LIBERO 评测里有大量和去噪步数无关的固定开销：

- 环境 `env.step`
- replan 窗口里的动作执行
- 任务/episode 调度
- 视频保存、日志、结果写盘
- 任务间的 Python / Hydra / multiprocessing 开销

这些开销在 `release_1` 和 `release_10` 里都存在，所以它们会把步数差异稀释掉。

## Q3: 为什么纯推理会到 `4.46x`，甚至去噪循环接近 `10x`？

因为 `experiments/libero/benchmark_release_inference_latency.py` 只测模型推理：

- 先固定好 `input_image`、`context`、`proprio`
- 先做一次 video KV cache prefill
- 然后只重复 action denoise loop

因此：

- `release_10` 比 `release_1` 多 9 次 action denoise
- 纯 denoise loop 自然接近 `10x`
- 但端到端里还包含固定前处理，所以会下降到 `4.46x`

## Q4: 这次正式结果放在哪里？

- 正式纯推理结果：[`evaluate_results/latency/release_1_vs_10_formal_gpu3.json`](../../evaluate_results/latency/release_1_vs_10_formal_gpu3.json)
- full LIBERO baseline summary：
  - [`evaluate_results/libero/release_baseline_20260512_steps1/summary.json`](../../evaluate_results/libero/release_baseline_20260512_steps1/summary.json)
  - [`evaluate_results/libero/release_baseline_20260512_steps10_v2/summary.json`](../../evaluate_results/libero/release_baseline_20260512_steps10_v2/summary.json)

## Q5: 应该怎么引用这个结论？

建议写成两句话：

1. `release_1` 和 `release_10` 在 full LIBERO 闭环评测上的总时间比约为 `1.55x`。
2. 纯模型推理层面，`infer_action` 的端到端时间比为 `4.46x`，纯去噪循环为 `9.77x`。

这样不会把闭环系统成本和模型本体成本混在一起。
