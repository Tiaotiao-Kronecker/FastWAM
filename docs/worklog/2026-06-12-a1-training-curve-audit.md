# A1 Training Curve Audit

日期：2026-06-12

## 背景

严格 350-trial gap-probe 已经确认 A1 70k 强于 full-state A1 80k continuation：

| 权重 | 结果 |
| --- | ---: |
| release10 | 338/350 = 96.57% |
| A1 70k | 333/350 = 95.14% |
| A1.3 full clip0.25 70.3k | 329/350 = 94.00% |
| A1.2 full residual-only 70.3k | 326/350 = 93.14% |
| A1 full-control 80k | 324/350 = 92.57% |
| release1 | 314/350 = 89.71% |

这个结论只证明 70k 优于已测试的 80k continuation 和几个续训分支，不证明 70k 是原始 A1 训练过程的全局最优 checkpoint。

## H200-1 检查结果

原始 A1 run：

```text
/DATA/disk2/wangchen/projects/FastWAM/runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000
```

可用权重：

```text
step_005000.pt
step_010000.pt
...
step_070000.pt
```

完整训练状态只找到：

```text
checkpoints/state/step_070000/
```

中间 optimizer/scheduler/random state 没找到。

WandB 离线历史：

```text
wandb/offline-run-20260529_174345-cbn0qyw0/run-cbn0qyw0.wandb
```

已解析出 7000 条 train history，匹配 `log_every=10` 到 70k。

H200-1 上已有的 `30k/70k` eval 目录：

```text
evaluate_results/libero/a1_lora_eqanchor_step030000_gap_probe_t0_20260530
evaluate_results/libero/a1_lora_eqanchor_step070000_gap_probe_t0_20260601
```

两者都在 `libero_goal,6` 上 `rc=134`，没有可用 `summary.csv`、`summary.json` 或 `task_success_rates.csv`，不能用于比较 30k 和 70k。

## 训练中间量

本地详细表：

```text
evaluate_results/libero/a1_training_curve_audit_20260612_metrics.csv
evaluate_results/libero/a1_training_curve_audit_20260612.md
```

注意：`evaluate_results/` 被 `.gitignore` 忽略，提交时需要 `git add -f`，或以本文档为可提交摘要。

Trailing 1k-window 摘要：

| Step | LR | meanflow loss | loss std | grad norm | dudt rms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30000 | 3.3009e-05 | 0.010448 | 0.006738 | 0.471199 | 0.267545 |
| 40000 | 2.1465e-05 | 0.008853 | 0.006765 | 0.407988 | 0.275539 |
| 50000 | 1.0750e-05 | 0.007561 | 0.005976 | 0.348819 | 0.247202 |
| 55000 | 6.4585e-06 | 0.013058 | 0.031307 | 0.380402 | 0.263367 |
| 60000 | 3.2109e-06 | 0.007799 | 0.007548 | 0.343849 | 0.237211 |
| 65000 | 1.1873e-06 | 0.006973 | 0.008367 | 0.315219 | 0.228132 |
| 70000 | 5.0000e-07 | 0.013004 | 0.064246 | 0.334760 | 0.249512 |

Rolling 1k minima：

| Metric | Best window end | Mean |
| --- | ---: | ---: |
| `train/loss_meanflow_target` | 54000 | 0.006346 |
| `train/loss` | 54000 | 0.007236 |
| `train/grad_norm` | 65000 | 0.315219 |
| `train/meanflow_dudt_rms` | 64000 | 0.226351 |

## 判读

A1 在 50k-65k 已经进入低 loss 平台区。多个 proxy 指标在 54k-65k 达到更好值，70k 的 trailing 1k loss 标准差明显更大，说明尾部有尖峰。因此不能断言 70k 是最优闭环 checkpoint。

但训练 loss 不是闭环成功率。尤其 one-step action policy 会出现 loss 与 rollout 脱钩，所以更早 checkpoint 是否优于 70k 必须补 strict eval。

建议下一步用同一 350-trial strict gap-probe 评：

```text
step_050000.pt
step_055000.pt
step_060000.pt
step_065000.pt
step_070000.pt
```

资源紧张时先评：

```text
step_060000.pt
step_065000.pt
step_070000.pt
```

协议必须和 `20260611_190935` strict run 保持一致：同 task set、seed、trial count、`policy_subprocess`、action trace 设置和 `num_inference_steps=1`。

## 关于 80k 退化

70k -> 80k 是 full-state continuation，但 scheduler 在 70k 已经到 cosine minimum。PyTorch `CosineAnnealingLR` 继续越过 `T_max` 后，LR 会反弹：

| State | Cosine last epoch | LR |
| --- | ---: | ---: |
| 70k | 66500 | 5.0000e-07 |
| 75k | 71500 | 1.1870e-06 |
| 80k | 76500 | 3.2109e-06 |

所以 80k 不是低学习率静止尾训，而是 post-minimum LR rebound 加 batch-1 stochastic updates。它足以把 one-step policy 从较好的闭环行为区推开。

## A1 checkpoint sweep 已完成

同日补跑了 A1 中间 checkpoint strict gap-probe sweep：

```text
RUN_STAMP=20260612_144300
GPU=7
task_file=experiments/libero/task_sets/libero_gap_probe_v1.txt
7 tasks x 50 trials = 350 trials/checkpoint
seed=42
trial_indices=null
num_inference_steps=1
policy_subprocess=false
save_action_trace=true
save_rollout_video=false
```

所有 checkpoint 均完成，`failed_tasks=0`。

详细归档：

```text
evaluate_results/libero/a1_ckpt_sweep_strict_gap_probe_gpu7_20260612_144300.md
evaluate_results/libero/a1_ckpt_sweep_strict_gap_probe_gpu7_20260612_144300_overall.csv
evaluate_results/libero/a1_ckpt_sweep_strict_gap_probe_gpu7_20260612_144300_task_breakdown.csv
scripts/run_strict_gap_probe_a1_ckpt_sweep_gpu7_20260612.sh
```

注意：`evaluate_results/` 被 `.gitignore` 忽略，提交时需要 `git add -f`，或以本文档为可提交摘要。

### Sweep 结果

| Checkpoint | Total | Overall | Spatial | Goal | LIBERO-10 subset | vs 70k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 step070000 | 333/350 | 95.14% | 49/50 | 98/100 | 186/200 | baseline |
| A1 step060000 | 327/350 | 93.43% | 49/50 | 97/100 | 181/200 | -6 |
| A1 step065000 | 327/350 | 93.43% | 50/50 | 97/100 | 180/200 | -6 |
| A1 step055000 | 323/350 | 92.29% | 47/50 | 99/100 | 177/200 | -10 |
| A1 step050000 | 319/350 | 91.14% | 48/50 | 92/100 | 179/200 | -14 |

Task-level successes out of 50:

| Checkpoint | spatial 3 | goal 5 | goal 6 | 10-0 | 10-2 | 10-3 | 10-8 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 step050000 | 48 | 48 | 44 | 43 | 50 | 50 | 36 | 319 |
| A1 step055000 | 47 | 49 | 50 | 42 | 49 | 50 | 36 | 323 |
| A1 step060000 | 49 | 48 | 49 | 44 | 49 | 49 | 39 | 327 |
| A1 step065000 | 50 | 48 | 49 | 44 | 49 | 49 | 38 | 327 |
| A1 step070000 | 49 | 48 | 50 | 45 | 49 | 50 | 42 | 333 |

### 修订判读

此前训练曲线提示 50k-65k 已进入低 loss 平台，因此不能仅凭训练指标断言 70k 是最佳。现在 strict rollout sweep 已补齐：在已保存并同口径评测的 50k/55k/60k/65k/70k 中，70k 是最佳 checkpoint。

70k 复测结果仍是 `333/350 = 95.14%`，与 `20260611_190935` strict eval 完全一致，说明 A1 70k baseline 在这套 eval wrapper 下稳定。

因此后续不应再围绕“是否改用更早 A1 checkpoint”展开。A1 70k 应固定为 Phase-1 attribution 起点。80k 退化应解释为 70k 之后的 continuation damage，尤其是 scheduler 越过 cosine minimum 后 LR 反弹叠加 batch-1 stochastic updates，而不是 70k 本身已经明显过训。

## MeanFlow 下一步

新的执行判断：

1. 固定 A1 70k 作为 attribution 起点和 baseline。
2. 不再投入 A1-continue-control 的普通 10k 延长训练作为性能候选；已测 full-state 80k 低于 70k。
3. 继续推进 A1.4/A2 的核心问题：是否通过 interval mixture、residual target 形式、clipping 或 endpoint 改善 one-step rollout，而不是靠多训。
4. 后续 continuation 需要避免原 scheduler 的 post-minimum LR rebound；若改变 LR/scheduler，则必须同步跑 control，保证 attribution 公平。
5. 评估 gate 应优先对准 hard tasks，而不是只看训练 loss。现有 gap probe 可继续作为便宜 readout，但 A2 归因需要补计划中的 hard/control probe。

## 2026-06-12 晚间续训可行性检查

时间：2026-06-12 21:39 CST。

### 70k 起点策略

用户提出疑问：既然 A1 70k 已经是已保存 checkpoint 中的局部较好点，且普通续训会让性能下降，为什么还从 70k 开始。

记录结论：

- 70k 仍然合理作为 Phase-1 attribution 起点，因为它是当前同口径 strict gap-probe 下最强 A1 起点。
- 后续 continuation 不能被解释为“继续多训 A1”，而是固定强起点测试新的 objective 因素，尤其是 interval mixture 是否修复 one-step rollout mismatch。
- plain A1 70k->80k 已经是负 control，不再作为性能候选。
- 若后续为了避免 post-minimum LR rebound 改 scheduler/LR，必须同步跑同 scheduler 的 control 和 mix，否则不能与旧 80k control 做严格 attribution。

### 机器资源

H200-1 `10-0-2-252`：

- 磁盘：`/DATA/disk2` 632G 可用，`/DATA/disk3` 142G 可用，`/` 560G 可用。
- GPU：8 张 H200 均有外部 `/DATA/disk8/maxliu/conda/envs/fastwam-robotwin-eval/bin/python` 进程占用，单卡显存约 30G 到 63G；当前没有干净 4-GPU 窗口。
- A1 state：存在 `/DATA/disk2/wangchen/projects/FastWAM/runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/checkpoints/state/step_070000`，约 26G，包含 `random_states_0..3.pkl` 和 4 个 ZeRO optimizer shard。
- 代码：已确认存在 `interval_sampling` runtime/model 支持。

H200-2 `10-0-2-144`：

- 磁盘：`/DATA/disk2` 715G 可用，`/DATA/disk3` 163G 可用，`/`/`/tmp` 169G 可用。
- GPU：GPU7 空闲；GPU0-6 有持续高 util 任务或 GPU 进程。
- A1 state：当前只有 `weights/step_070000.pt`，缺完整 `checkpoints/state/step_070000`。full-state continuation 需要先从 H200-1 同步约 26G state。

### 执行建议

1. 先把 A1 70k `checkpoints/state/step_070000` 从 H200-1 同步到 H200-2；H200-2 空间足够。
2. 在 H200-2 GPU7 做短 resume feasibility smoke，验证单卡是否能加载 4-rank ZeRO state；该 smoke 只用于工程可行性，不作为 official attribution 结果。
3. official full-state continuation 优先等待 4-GPU 窗口，使用与 A1 state 匹配的 ZeRO world size。
4. 首个 official 训练任务是 `A1.4-full-mix 70k->70.3k smoke`；通过后跑 `70k->80k`。
5. `A1.4-full-mix+clip` 只在 mix smoke 显示 residual norm 过大、NaN/loss 抖动或 clip 指标需要约束时投入完整 10k。

### H200-2 state 同步与单卡探针

已从 H200-1 同步 A1 70k full state 到 H200-2：

```text
runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/
  a1_lora_eqanchor_sync_20260529_174000/checkpoints/state/step_070000
```

同步后大小 `26G`，关键 shard 大小与 H200-1 一致：

```text
mp_rank_00_model_states.pt                         26926553024
bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt       33671877
bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt       33648517
bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt       33645765
bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt       33645829
```

`trainer_state.json`：

```json
{
  "global_step": 70000,
  "epoch": 1,
  "batch_in_epoch": 571
}
```

H200-2 的 `./data/libero_mujoco3.3.2` 只是空壳目录；真实训练数据在：

```text
/DATA/disk0/shared/datasets/libero_mujoco3.3.2/
```

第一次 feasibility smoke 失败于数据路径，第二次用绝对 `data.train.dataset_dirs` override 后进入 `accelerator.load_state()`，但 DeepSpeed 明确报错：

```text
The checkpoint being loaded used a DP world size of 4 but the current world size is 1.
Automatic adjustment of ZeRO's optimizer state partitioning with a new world size is not currently supported.
```

结论：

- H200-2 已具备 state 和数据路径前置条件。
- 单卡不能恢复这个 A1 70k ZeRO state；official full-state continuation 必须用 4-GPU world size。
- 当前 H200-2 只有 GPU6/GPU7 看起来低占用，不足 4 张；H200-1 所有 GPU 仍被外部 eval/python 进程占用约 60G 显存。因此此刻不能安全启动 official `A1.4-full-mix` 续训。

## 2026-06-13 H200-2 抢占式 4-GPU A1.4-full-mix 启动

用户决定不继续等待干净窗口，直接在 H200-2 与其他任务竞争 4 张卡启动 `A1.4-full-mix`。

### 4-GPU smoke

启动脚本：

```text
scripts/run_a1_4_full_mix_smoke_h2002_4gpu_20260612.sh
```

最终成功 run：

```text
tmux: a1_4_mix_h2002_4gpu_smoke_retry2_20260613
log: /tmp/a1_4_mix_h2002_4gpu_smoke_retry2_20260613.log
run_id: a1_4_full_mix_from070k_to70300_smoke_h2002_4gpu_retry2_20260613_1432
gpu_ids: 0,1,4,7
resume: runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/checkpoints/state/step_070000
data: /DATA/disk0/shared/datasets/libero_mujoco3.3.2
text_cache: /DATA/disk0/shared/datasets/text_embeds_cache/libero
```

结果：正常跑到 `70300/70300` 并保存：

```text
runs/libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5/
  a1_4_full_mix_from070k_to70300_smoke_h2002_4gpu_retry2_20260613_1432/
    checkpoints/weights/step_070300.pt
    checkpoints/state/step_070300
```

smoke 观察：

- 4-GPU full-state resume 成功，DeepSpeed world size 与 A1 70k state 匹配。
- interval mixture 指标正常出现：`meanflow_interval_mode_e2e/local/random` 均有采样。
- `meanflow_clip_fraction=0`，符合 mix-only 不启用 clip 的预期。
- 末段 loss 和 residual 指标稳定；例如 `step=70300` 时 `loss=0.0023`、`meanflow_residual_rms=0.0466`、`lr=5.02e-07`。
- 竞争环境下速度约 `0.26-0.29 step/s`，即约 `1.05-1.15 samples/s`。

注意：虽然 smoke 保存了 `step_070300` full state，但该 state 的 `scheduler.bin` 来自 `max_steps=70300` 的 smoke 配置。正式 10k attribution run 不从该 state 继续，以避免把 smoke 的短周期 scheduler 状态带入正式 80k 训练。

### 正式 10k pilot

新增正式启动脚本：

```text
scripts/run_a1_4_full_mix_h2002_4gpu_20260613.sh
```

启动命令：

```text
FASTWAM_GPU_IDS=0,1,4,7 \
MASTER_PORT=29524 \
RUN_ID=a1_4_full_mix_from070k_to80000_h2002_4gpu_20260613_1508 \
bash scripts/run_a1_4_full_mix_h2002_4gpu_20260613.sh
```

运行位置：

```text
tmux: a1_4_mix_h2002_4gpu_full_20260613
log: /tmp/a1_4_mix_h2002_4gpu_full_20260613.log
output: runs/libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5/a1_4_full_mix_from070k_to80000_h2002_4gpu_20260613_1508
```

正式 run 直接从 A1 70k 原始完整 state 启动：

```text
runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/
  a1_lora_eqanchor_sync_20260529_174000/checkpoints/state/step_070000
```

配置要点：

```text
task=libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5
max_steps=80000
save_every=5000
save_training_state=true
eval_every=0
log_every=10
data.train.dataset_dirs=/DATA/disk0/shared/datasets/libero_mujoco3.3.2/{libero_spatial,libero_object,libero_goal,libero_10}_no_noops_lerobot
data.train.text_embedding_cache_dir=/DATA/disk0/shared/datasets/text_embeds_cache/libero
```

启动后已确认：

- `All model weights loaded successfully`
- `All optimizer states loaded successfully`
- `All scheduler states loaded successfully`
- `All dataloader sampler states loaded successfully`
- `All random states loaded successfully`
- `Restored dataloader progress: epoch=1 batch_in_epoch=571 sample_offset=2284`
- `Starting training with max_steps=80000`

首批训练日志已出现，说明正式 10k pilot 已排上并开始更新：

```text
step=70010/80000 loss=0.0060 meanflow_residual_rms=0.0706 lr=5.00e-07 speed=0.27 step/s eta=10:23:06
step=70020/80000 loss=0.0190 meanflow_residual_rms=0.1310 lr=5.00e-07 speed=0.27 step/s eta=10:07:26
step=70030/80000 loss=0.0116 meanflow_residual_rms=0.1016 lr=5.00e-07 speed=0.28 step/s eta=09:59:13
step=70040/80000 loss=0.0102 meanflow_residual_rms=0.0890 lr=5.00e-07 speed=0.28 step/s eta=09:58:56
```

当前判读：正式 `A1.4-full-mix 70k->80k` 已在 H200-2 以 4-GPU world size 运行。它是 old-scheduler protocol 下的 mix attribution run，可与既有 old-scheduler `A1-full-continue-control 80k` 做第一层比较；若后续切换 tail-scheduler protocol，则仍需同步重跑 tail-control。
