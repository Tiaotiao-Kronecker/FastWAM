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

1. 保留正在运行的 `A1-continue-control`。
2. 实现并 smoke `A1.2-residual-only`；通过后在空闲单卡启动 10k。
3. 实现并 smoke `A1.3-clip`；若 residual-only 前 100 step 稳定，可并行启动 10k。
4. `A1.4-mix` 等 sampler 实现后再并行，不与 endpoint 或数据上采样混开。

执行记录：

- `A1.3-clip` 初始 `token_l2 max_norm=2.0` smoke 全程 `clip_fraction=0`，说明阈值对当前 residual scale 过松。
- A1 attribution 阶段将 `A1.3-clip` 暂定为 `token_l2 max_norm=0.25`，并新增 residual token norm 日志；若 smoke clip fraction 过高再调到 `0.5`。

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
