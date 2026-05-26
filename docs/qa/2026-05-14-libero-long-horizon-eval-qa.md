# 2026-05-14 LIBERO Long-Horizon Eval QA

## Q1: 为什么要额外建立 LIBERO 长程 eval 集合？

当前 full LIBERO baseline 太接近饱和：

| Setting | Success |
| --- | ---: |
| `release_1` | 95.80 |
| `release_10` | 96.80 |

`release_10 - release_1` 只有 +1.00 point。这个差距太小，不适合判断 one-step 训练目标是否真正改善了大步 action flow。

## Q2: 我们能不能直接“造更长”的 LIBERO 任务？

不建议把它作为第一步。

官方 LIBERO benchmark 的任务和 initial states 是固定的。直接修改任务定义会引入 benchmark drift，结果不再能和论文或现有 baseline 对齐。

更稳妥的做法是：先从官方任务中抽出更长程、更不饱和的标准任务集合。

## Q3: 新增了哪些 task set？

文件位于 `experiments/libero/task_sets/`。

| Set | 用途 | Tasks | release_1 | release_10 | Gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `libero_long_official` | 官方 `libero_10` 长程 readout | 10 | 92.20 | 94.60 | +2.40 |
| `libero_long_horizon_v1` | 主要开发集，扩大 one-step 空间 | 13 | 91.85 | 95.54 | +3.69 |
| `libero_gap_probe_v1` | 快速诊断，按已有 gap 选取 | 7 | 89.71 | 96.57 | +6.86 |

## Q4: 哪个结果可以正式报告？

`libero_long_official` 最干净，因为它就是官方 `libero_10` 全 10 个任务。

`libero_long_horizon_v1` 可以作为开发集，但需要注明它是为 one-step 诊断构造的标准任务子集。

`libero_gap_probe_v1` 只能做内部 debug，因为它使用已有 baseline gap 进行选择，有明显 selection bias。

## Q5: 代码上改了什么？

1. 修正 `experiments/libero/run_libero_manager.py`：现在传入 `MULTIRUN.task_file` 时会真正使用已有 task file，不再覆盖成 full suite。
2. 新增 `experiments/libero/run_libero_task_set_eval.sh`：统一运行自定义 task set。
3. 新增 `experiments/libero/task_sets/*.txt`：保存长程/困难集合。
4. 新增 `docs/plan/2026-05-14-libero-long-horizon-eval-plan.md`：归档设计动机、集合定义和运行命令。

## Q6: 推荐怎么跑？

开发时先跑：

```bash
NUM_INFERENCE_STEPS=1 \
TASK_SET=libero_long_horizon_v1 \
experiments/libero/run_libero_task_set_eval.sh
```

然后用同一集合跑 10-step 对照：

```bash
NUM_INFERENCE_STEPS=10 \
TASK_SET=libero_long_horizon_v1 \
RUN_TAG=libero_long_horizon_v1_release10 \
experiments/libero/run_libero_task_set_eval.sh
```

正式长程 readout 优先跑：

```bash
NUM_INFERENCE_STEPS=1 \
TASK_SET=libero_long_official \
experiments/libero/run_libero_task_set_eval.sh
```
