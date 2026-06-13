# FastWAM MeanFlow A2 Design Plan

日期：2026-06-03

## 背景

A1 Action-LoRA 已证明冻结 video/proprio、只训练 action 侧受控容量不是死路。A1 70k 整体表现可用，但 LIBERO-10 task 4/6/9 仍弱，问题集中在多物体空间关系、精确放置和长程末端修正。训练 loss 已很低，因此下一步不应只增加迭代次数，而应调整 MeanFlow 训练目标和 interval 分布。

## 核心判断

- TwinFlow 的主 residual 目标本质是无显式 JVP 的 MeanFlow：用有限差分逼近 `d/dt[(t-r)u]=v`。
- TwinFlow 的 fake/adv branch 是额外分布匹配机制，不属于标准 MeanFlow；FastWAM A2 暂不采用。
- A2 不作为一次性大改实验；必须拆成可归因的 ablation ladder。
- A2 优先验证 interval mixture、finite-difference residual target 和 residual clipping。
- Equal-time anchor 与 MeanFlow 不矛盾；它是 \(r=t\) 极限下对 release instantaneous velocity 的边界约束。

## A2 方案

- 从 release checkpoint clean retrain。
- 冻结 video expert。
- 冻结 proprio encoder。
- 训练 interval conditioner、action head、action LoRA/adapter。
- 默认 `conditioner_mode=joint_delta`。
- A2-core 保持 A1 的 LoRA rank 4、target modules、数据分布和主要超参。
- LoRA rank 8 只进入后续 `A2-capacity`。
- 不引入 fake/adv branch，不使用负时间 fake domain。

## Ablation Ladder

- `A1-control`：不改动，复现或直接使用 A1 70k。
- `A2-clip`：只加 residual clipping；interval sampling 仍为 A1 random。
- `A2-mix`：只加 interval mixture；residual 不 clip。
- `A2-core`：interval mixture + residual clipping；其他保持 A1 不变。
- `A2-endpoint`：在 A2-core 上单独加 endpoint loss。
- `A2-data`：在已验证 objective 上单独对 LIBERO-10 task 4/6/9 做 2x 到 3x 上采样。
- `A2-capacity`：在已验证 objective/data 上 clean retrain rank 8 和更多 LoRA target modules。

## 起点选择

两条线都需要，但回答的问题不同。

### Phase 1: A1-continue attribution

起点：A1 70k checkpoint。

目的：便宜地筛选 A1 差在哪里，以及哪个 A2 改动能修。

限制：

- 保持 LoRA rank 4。
- 保持 A1 target modules。
- 保持数据分布和 eval set。
- 不改 LR 策略，除非 control 和所有变体同步改。
- 必须有 `A1-continue-control`，排除“只是多训了”的收益。

桥接实验：

- `A1-continue-control`：A1 70k 继续训，仍用 A1 原 loss/random interval/endpoint=0。
- `A1.1-codepath`：走新代码路径，但配置等价 A1，用来查实现偏差。
- `A1.2-residual-only`：只把 target 写成 residual 形式，不开 clipping/mixture/endpoint。
- `A1.3-clip`：只加 residual clipping。
- `A1.4-mix`：只加 interval mixture；建议同时保留 mix-only 和 mix+clip。
- `A1.5-endpoint`：在胜出的最小 objective 上加 endpoint loss，先 `0.02`，再 `0.05`。

### Phase 2: Release-clean validation

起点：release flow-matching checkpoint。

目的：判断 A2 作为新 recipe 是否真的优于 A1。

实验：

- `A1-r4-clean`：同环境复现 A1。
- `A2-r4-clean`：从 release 训练 A1-continue 胜出的最小 objective，保持 rank 4。
- `A2-r4-data`：单独加 task 4/6/9 上采样。
- `A2-r8-clean`：单独测试 rank 8 / expanded target modules。

## Loss

令 `F_t = u_theta(x_t,r,t)`，`I_t=(t-r)F_t`。

有限差分残差：

```text
x_{t-delta} = x_t - delta * v_t
R = ((t-r)F_t - (t-delta-r)F_{t-delta}) / delta - v_t
```

clipped target：

```text
R_clip = clip(R, c)
y_mf = stopgrad(F_t - R_clip)
L_mf = mse(F_t, y_mf)
```

总 loss：

```text
L_A2 = 1.0 * L_mf
     + 0.20 * L_equal_time_velocity
     + 0.00 * L_action_endpoint
```

其中 equal-time anchor 可用 `equal_time_anchor_prob=0.25` 采样。Endpoint loss 默认关闭；若 A2-core 胜出，再单独测试 `0.02/0.05`。

## Interval Mixture

初始比例：

- `e2e`: 0.30，`r=0,t=1`，可选 `e2e_jitter=0.02`。
- `local`: 0.30，`delta=t-r` 在 `[0.02,0.15]`。
- `random/any`: 0.40，随机 `0<=r<t<=1`，保留最小 interval。

## 训练计划

- 数据：full LIBERO no-noops four-suite。
- A2-core 不改变数据权重。
- 困难任务上采样只放入 `A2-data`。
- Smoke：100 到 300 steps，检查 NaN、clip fraction、mode-wise loss。
- Pilot：10k A1-continue，跑 `A1-continue-control`、`A1.3-clip`、`A1.4-mix`、`A1.4-mix+clip`，先评估 hard task mini-set 和 control tasks。
- Mid：30k A1-continue，只保留 10k 胜出的 1 到 2 个分支。
- Main：70k release-clean，跑 `A1-r4-clean`、`A2-r4-clean`，再逐项追加 endpoint/data/capacity。

## 阶段性并行计划

### 可以并行

这些实验都是从同一个 A1 70k 权重出发，不依赖彼此 checkpoint，只要各自 smoke 通过即可分别占一张单卡跑 10k：

- `A1-continue-control`：当前控制组，继续 A1 原 finite-difference target。
- `A1.2-residual-only`：新增 residual target 代码路径，clip 关闭，验证与 A1 FD target 等价。
- `A1.3-clip`：在 residual-only 上只打开 token-L2 residual clipping。
- `A1.4-mix`：只改 interval sampling。
- `A1.4-mix+clip`：interval mixture 与 clipping 的最小组合。

### 需要串行或半串行

- `A2-r4 combined`：等待 10k attribution 结果，选择有效因素组合。
- `A2-r8 clean`：等待 r4 combined 有希望后再扩大 LoRA 容量。
- `Release-clean validation`：等待 A1-continue 阶段筛出最小有效 objective 后再从 release 权重重训。

### 当前执行顺序

截至 2026-06-13，Phase 1 已切回严格 full-state continuation 协议，不再把 weight-only finetune 结果当成 A1 70k 续训结论。70k 是旧 objective 下已保存 checkpoint 的最佳起点；后续只用它做新 objective 和 scheduler 归因。

1. `A1-full-continue-control 70k->80k` 已完成且低于 A1 70k；保留为 old-scheduler control。
2. `A1.2-full-residual-only` 已完成 70k->70.3k sanity；只作为 residual 代码路径检查，不投入完整 10k。
3. `A1.4-full-mix` 是当前主线；70k->70.3k smoke 已完成，正式 4-GPU 70k->80k pilot 已启动。
4. `A1.4-full-mix-clip` 只在 mix smoke 或 pilot 显示 residual norm 过大、NaN/loss 抖动，或 clip 指标显示需要约束时加入。
5. `A1.3-full-clip0.25` 暂不单独投入 10k full-state 续训。
6. 若要避免 70k 后 cosine rebound，必须使用成对 tail-scheduler protocol：同一 scheduler 下同步跑 control 与 mix；不能直接拿旧 `A1-full-continue-control 80k` 做严格对照。

执行记录：

- `A1.3-clip` 初始 `token_l2 max_norm=2.0` smoke 全程 `clip_fraction=0`，说明阈值对当前 residual scale 过松。
- A1 attribution 阶段将 `A1.3-clip` 暂定为 `token_l2 max_norm=0.25`，并新增 residual token norm 日志；若 smoke clip fraction 过高再调到 `0.5`。

## 10k weight-only 结果后的缩减计划

H200-2 上已完成的三支 `step_010000.pt` 来自 `weights/step_070000.pt` 的 weight-only finetune，不是恢复 optimizer/scheduler/random state 的严格续训。它们不能替代 full-state continuation，但可以用于减少下一轮候选。

重合 7 个 gap-probe task 上：

| 方案 | Overall | LIBERO-10 | 判读 |
| --- | ---: | ---: | --- |
| A1 70k overlap baseline | 95.14% | 93.00% | 70k 参考值。 |
| control +10k | 91.43% | 89.50% | weight-only 多训本身退化，说明 optimizer/scheduler reset 是强干扰。 |
| residual-only +10k | 91.43% | 86.50% | 没有优于 control，且 LIBERO-10 更差；后续只作为代码路径 sanity。 |
| clip0.25 +10k | 90.57% | 86.00% | 单独 clip 没有收益；不再单独投入 full-state 10k。 |

下一轮 full-state continuation 保留：

| 优先级 | 方案 | 是否恢复完整 A1 70k state | 步数 | 状态 |
| --- | --- | --- | --- | --- |
| 既有 control | `A1-full-continue-control` | 是，使用 `checkpoints/state/step_070000` | `70000 -> 80000` | 已完成；旧 scheduler 下负 control。只有改 tail scheduler 时才重跑。 |
| 已完成 sanity | `A1.2-full-residual-only` | 是 | `70000 -> 70100/70300` | 已完成；检查 residual 代码路径，不作为性能候选。 |
| 主候选 | `A1.4-full-mix` | 是 | `70000 -> 80000` | 70k->70.3k smoke 已完成；正式 4-GPU 10k pilot 正在 H200-2 运行。 |
| 条件候选 | `A1.4-full-mix-clip` | 是 | `70000 -> 80000` | 仅当 mix smoke 不稳或 residual norm 过大时加入。 |
| 暂缓 | `A1.3-full-clip0.25` | 是 | 暂不跑 10k | 现有 10k weight-only 结果不支持单独投入资源。 |

Full-state continuation 的 `max_steps` 使用全局 step：`80000` 表示从 `70000` 续训 10k，不是从头训练 80k。启动方式必须匹配 A1 70k 的 ZeRO/DeepSpeed state；优先使用同等 4-GPU DeepSpeed/Accelerate 配置，避免再次触发单进程加载 ZeRO state 失败。

## A1 checkpoint sweep 后的执行修订

2026-06-12 在 H200-2 上补跑了 A1 原始训练中间 checkpoint sweep，使用同一 strict 350-trial gap-probe：

```text
RUN_STAMP=20260612_144300
task_file=experiments/libero/task_sets/libero_gap_probe_v1.txt
7 tasks x 50 trials
seed=42
num_inference_steps=1
policy_subprocess=false
save_action_trace=true
save_rollout_video=false
```

结果：

| Checkpoint | Overall | LIBERO-10 subset | 判读 |
| --- | ---: | ---: | --- |
| A1 step070000 | 95.14% | 93.00% | 已保存 checkpoint 中最佳；复测与 `20260611_190935` 完全一致。 |
| A1 step060000 | 93.43% | 90.50% | 低于 70k。 |
| A1 step065000 | 93.43% | 90.00% | 低于 70k。 |
| A1 step055000 | 92.29% | 88.50% | 低于 70k。 |
| A1 step050000 | 91.14% | 89.50% | 低于 70k。 |

结论：

- A1 70k 不是误选；在已保存的 50k/55k/60k/65k/70k 中是最佳 gap-probe checkpoint。
- 训练曲线在 50k-65k 进入低 loss 平台，但这没有转化为更好的闭环 rollout。
- A1 80k 退化应视作 70k 之后的 continuation damage，而不是 70k 已经明显过训。已确认的机制风险是 scheduler 越过 cosine minimum 后 LR 反弹，加上 batch-1 随机更新。

修订后的下一步：

1. 固定 A1 70k 作为 Phase-1 attribution 起点，不再寻找更早 A1 checkpoint。
2. 不再把普通 `A1-full-continue-control 70k->80k` 当性能候选；它现在主要作为“多训会退化”的 control 证据。
3. 优先实现并启动 `A1.4-full-mix` smoke/10k pilot。它直接验证 interval 分布和 one-step rollout 匹配是否是主问题。
4. 同步准备 `A1.4-full-mix-clip`，但只有在 mix smoke 显示 residual norm/clip 指标需要约束，或 mix-only 不稳时再投入完整 10k。
5. 若后续 continuation 改 scheduler 或 LR 以避免 post-minimum rebound，必须同步跑 control 或把已有 80k control 标记为“旧 scheduler control”，否则不能做严格 attribution。
6. endpoint、data upweight、rank8、release-clean 仍等待 attribution winner；没有 mix/mix+clip 胜出前不推进。

## 2026-06-12 续训执行计划与机器可行性

### 为什么仍从 A1 70k 出发

A1 70k 是已保存 A1 checkpoint 中同口径 strict gap-probe 的最佳点，但这只说明旧 A1 objective 在 70k 附近更好，不说明新 objective 不能从这个强起点继续做归因。后续 continuation 的含义不是“多训 A1”，而是固定最强起点，最小化变量，测试 interval mixture 是否能修复 one-step rollout mismatch。

因此后续判断标准改为：

- 不再把 plain A1 70k->80k 当性能候选。
- `A1.4-full-mix` 必须相对 old-scheduler control 有改善，并尽量不低于 A1 70k 的 hard/control task 表现。
- 如果改变 LR 或 scheduler 来避免 post-minimum rebound，必须同步跑新的 control 与 mix；旧 80k control 只能作为旧 scheduler 证据。

### 推荐执行协议

1. Preflight：统一 H200-1/H200-2 代码 commit，确认 `interval_sampling` 与 `residual_clip` 实现一致；记录 git head、dirty diff、数据路径、A1 70k state 路径。
2. State：必须使用 `checkpoints/state/step_070000`，不是 `weights/step_070000.pt`。该 state 含 4 个 ZeRO optimizer shard，official full-state continuation 优先使用同等 4-GPU world size。
3. Scheduler：先明确采用哪种 protocol：
   - old-scheduler protocol：完整恢复 scheduler，直接比较现有 `A1-full-continue-control 80k` 与新的 `A1.4-full-mix 80k`；若 mix 失败，结论会被 LR rebound 混淆。
   - tail-scheduler protocol：恢复 model/optimizer/random state 后重置 70k 后 tail LR，避免 cosine rebound；必须同步跑 `tail-control` 和 `tail-mix`，不能使用旧 control 做严格对照。
4. Smoke：`A1.4-full-mix` 先跑 `70000 -> 70100/70300`，检查 LR、scheduler last_epoch、mode fraction、NaN、loss 抖动、`meanflow_residual_rms`、`meanflow_residual_token_norm_*`。
5. Pilot：smoke 稳定后跑 `70000 -> 80000`。`A1.4-full-mix+clip` 只在 mix smoke 显示 residual norm 过大或不稳定时投入完整 10k。
6. Eval gate：使用同一 strict 350-trial gap-probe，加计划中的 hard/control attribution set。若 mix 或 mix+clip 没有提升 hard tasks，endpoint、data upweight、rank8、release-clean 全部继续暂缓。

### 2026-06-12 21:39 CST 资源快照

| 机器 | 磁盘 | GPU 状态 | A1 70k state / 代码 | 续训可行性 |
| --- | --- | --- | --- | --- |
| H200-1 `10-0-2-252` | `/DATA/disk2` 632G 可用，`/DATA/disk3` 142G 可用，`/` 560G 可用。 | 8 张 H200 均有外部 python/eval 进程占用，单卡显存约 30G 到 63G；当前没有干净 4-GPU 窗口。 | 有 `checkpoints/state/step_070000`，约 26G，含 `random_states_0..3` 与 4 个 ZeRO optimizer shard；已确认有 interval mixture 支持。 | 磁盘与 state 最完整，但当前 GPU 不适合启动 official full-state 10k。等 4 张卡空出后是最省同步成本的选择。 |
| H200-2 `10-0-2-144` | `/DATA/disk2` 715G 可用，`/DATA/disk3` 163G 可用，`/`/`/tmp` 169G 可用。 | GPU7 空闲；GPU0-6 有持续高 util 任务或进程占用。 | 当前只有 A1 70k `.pt` 权重，缺 `checkpoints/state/step_070000`；需要从 H200-1 同步约 26G state。 | 磁盘足够，GPU7 可做单卡 load/smoke 探针；official 4-rank full-state 续训需先同步 state，并等待至少 4 张合适 GPU。 |

当前推荐：优先把 A1 70k full state 同步到 H200-2，因为空间足够且 GPU7 已空；随后在 GPU7 做“resume feasibility smoke”来验证单卡能否加载 4-rank ZeRO state。若单卡 resume 失败或不作为 official 口径，则等待 H200-2 或 H200-1 出现 4-GPU 窗口，再启动 official `A1.4-full-mix` smoke/10k。

更新：A1 70k full state 已同步到 H200-2，单卡 feasibility smoke 已验证不可行。DeepSpeed 报错指出 checkpoint 的 DP world size 是 4，而当前 world size 是 1，ZeRO optimizer state 不能自动重分片。因此 official `A1.4-full-mix` 必须等待 4-GPU window；H200-2 训练数据需用 `/DATA/disk0/shared/datasets/libero_mujoco3.3.2` 绝对路径或等价 symlink。

### 2026-06-13 更新：H200-2 抢占式 4-GPU 已启动

用户决定在 H200-2 直接竞争 4 张卡启动 A1.4。执行记录：

- `A1.4-full-mix 70k->70.3k smoke` 已完成，4-GPU full-state resume、数据路径、text cache 均验证通过。
- smoke 输出：
  `runs/libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5/a1_4_full_mix_from070k_to70300_smoke_h2002_4gpu_retry2_20260613_1432/checkpoints/weights/step_070300.pt`
- smoke 末段指标稳定，`meanflow_clip_fraction=0`，interval mixture 的 `e2e/local/random` 采样均出现。
- 正式 `A1.4-full-mix 70k->80k` 已从原始 A1 70k full state 重新启动，而不是接 `70300` smoke state，以避免使用 smoke 的 `max_steps=70300` scheduler 状态。

正式 pilot：

```text
tmux: a1_4_mix_h2002_4gpu_full_20260613
log: /tmp/a1_4_mix_h2002_4gpu_full_20260613.log
run_id: a1_4_full_mix_from070k_to80000_h2002_4gpu_20260613_1508
script: scripts/run_a1_4_full_mix_h2002_4gpu_20260613.sh
gpu_ids: 0,1,4,7
resume: runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/checkpoints/state/step_070000
max_steps: 80000
save_every: 5000
```

已确认 `model/optimizer/scheduler/dataloader/random states` 全部恢复成功，并开始训练到 `step=70040/80000`。当前速度在竞争环境下约 `0.27-0.28 step/s`，ETA 约 10 小时。到 `75000` 会保存中间 full state，到 `80000` 保存最终 weights/state。

下一步 gate：

1. 等待 `step_080000.pt` 与 `state/step_080000` 落盘。
2. 用 strict 350-trial gap probe 评估 `A1.4-full-mix 80k`。
3. 首先与 A1 70k baseline 和 old-scheduler `A1-full-continue-control 80k` 对比。
4. 只有 mix 在 hard tasks 上有明确收益且 controls 不明显退化，才进入 30k continue 或考虑 endpoint/data/rank8。
5. 若 mix-only 出现 residual norm 或稳定性问题，再启动 `A1.4-full-mix+clip`；当前 smoke 不支持立即追加 mix+clip 完整 10k。

## 监控指标

- `loss/mf_e2e`
- `loss/mf_local`
- `loss/mf_random`
- `meanflow/residual_norm`
- `meanflow/clip_fraction`
- `loss/equal_time_velocity`
- `loss/action_endpoint`
- `params/lora_norm`
- `grads/action_head_norm`

## 通过标准

- LIBERO-10 task 4/6/9 相比 A1 提升。
- A1 强项 task 不显著退化。
- clip fraction 早期可高，但不应长期接近 100%。
- 不以训练 loss 接近 0 作为成功标准。

详细 HTML 归档见 `docs/meanflow_action_a2_design.html`。
