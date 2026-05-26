# ZipMotion / ZipMo + FastWAM 性能评估方案

日期：2026-05-14

## 目标

这份方案回答一个问题：ZipMotion / ZipMo motion tokens 接入 FastWAM 后，怎样判断它是真的提升了策略能力，而不是只提升了某个离线 loss，或者引入了未来信息泄漏。

总目标分三层：

```text
1. Motion 表示是否有效
2. Motion 条件是否被 FastWAM 用上
3. 闭环机器人任务成功率、鲁棒性和效率是否提升
```

最终报告必须以闭环 success rate 为主指标；open-loop action error、track prediction error、video PSNR/SSIM 只能作为诊断指标。

## 被评估的候选方案

### A0：FastWAM 原始 baseline

不接 ZipMotion。作为所有结果的锚点。

```text
input = image + language + proprio
policy = FastWAM release or fine-tuned FastWAM
```

### A1：Motion-token oracle 上界

只用于诊断，不可作为正式 policy 结果。

```text
input = current obs + future GT tracks encoded motion tokens
```

目的：

- 判断 motion token 通路是否有能力影响 action。
- 给出一个“如果 motion 完全准，最多可能提升多少”的上界。

硬性标注：

```text
oracle uses future information, not deployable.
```

### A2：Predicted Motion Context Adapter

第一版正式方案。

```text
current / history frames
  -> ZipMotion predictor
  -> predicted motion tokens
  -> Linear projection to FastWAM context dim
  -> concat(text, proprio, motion)
  -> FastWAM action generator
```

训练时可冻结 ZipMotion，只训练 motion projection + action branch / adapter。

### A3：External-only / Wrist-only / Multi-view Motion

用于判断多相机 motion 设计是否合理。

```text
FastWAM + external-only motion
FastWAM + wrist-only motion
FastWAM + external主干 + wrist gated residual
FastWAM + external+wrist naive concat
```

### A4：Gated Cross-Attention Adapter

第二阶段方案。

```text
h_action = h_action + sigmoid(g) * CrossAttn(h_action, z_motion)
```

只有 A2 证明 motion context 有效后再做。

## 评估阶段

## Phase 0：信息泄漏检查

这是所有实验前的必要检查。

必须确认：

```text
训练输入可以包含：
  - 当前帧 / 历史帧
  - 当前 proprio
  - language
  - 由当前/历史帧预测出的 motion tokens

训练 target 可以包含：
  - 未来 tracks
  - future motion latent
  - future video
  - ground-truth action

eval / deployment 输入不允许包含：
  - future GT frames
  - future GT tracks
  - 由完整 episode 未来帧编码出的 motion latent
```

检查项：

| 检查 | 判定 |
|---|---|
| motion tokens 是否由当前/历史观测生成 | 必须是 |
| dataloader 是否把 future tracks 放进 policy input | 必须否 |
| oracle 实验是否单独标注 | 必须是 |
| eval 脚本是否禁用 GT future tracks input | 必须是 |

如果这一步失败，后续 success rate 没有意义。

## Phase 1：ZipMotion 自身 motion 预测评估

目的：先确认 ZipMotion 在机器人数据上的 motion 表示质量。否则 FastWAM 失败时无法判断是 motion token 差，还是融合方式差。

### 数据集

优先顺序：

```text
LIBERO / RoboTwin sanity subset
DROID held-out scenes
Bridge held-out tasks
自采真机 small validation
```

### 指标

#### Track reconstruction / prediction

对可见点计算：

```text
EPE = mean || p_pred(t) - p_gt(t) ||_2
PCK@5px  = percent(EPE < 5)
PCK@10px = percent(EPE < 10)
visibility BCE / accuracy
```

按 horizon 分开统计：

```text
short: 1-5 frames
mid:   6-15 frames
long:  16+ frames
```

#### Action relevance

训练一个轻量 probe：

```text
z_motion -> action chunk
```

指标：

```text
action L1 / L2
EEF position error
gripper open/close accuracy
```

这个 probe 不替代 policy，只用于判断 motion latent 是否包含 action-relevant 信息。

#### View robustness

对 motion predictor 做相机 dropout：

```text
external only
wrist only
external + wrist
external dropped
wrist dropped
```

指标仍是 EPE / PCK / action probe error。

### 通过门槛

继续接 FastWAM 的最低条件：

```text
predicted motion tokens 明显优于 no-motion / random-motion probe
external+wrist 不显著差于 external-only
motion token 对 action probe 有可测信息增益
```

## Phase 2：Open-loop FastWAM 诊断

目的：在不跑昂贵闭环仿真前，确认 motion context 是否真的改变和改善 action 预测。

### 数据

使用 held-out validation episodes，不使用训练 episode。

### 指标

#### Action prediction

```text
action L1
action L2
EEF delta position L2
EEF rotation error
gripper BCE / accuracy
chunk endpoint error
chunk smoothness error
```

建议重点看：

```text
endpoint error = || a_pred[-1] - a_gt[-1] ||
```

因为控制里 action chunk 最后几步偏差通常会影响 replan 前状态。

#### Diffusion / flow 诊断

如果还是 diffusion policy：

```text
num_inference_steps = 1, 2, 4, 10
```

分别看 action error。目标不是只在 10-step 好，而是判断 motion 是否改善低步数采样。

#### Motion usage 检查

做三种输入：

```text
real predicted motion tokens
zero motion tokens
shuffled motion tokens
```

如果三者 action error 几乎一样，说明 FastWAM 可能没有用上 motion。

### 通过门槛

建议进入闭环 eval 的条件：

```text
predicted motion tokens 比 zero/shuffled tokens 降低 action endpoint error
至少在部分 task family 上 open-loop action error 改善
低步数推理不显著退化
```

注意：open-loop 改善不保证闭环成功率提升，但 open-loop 完全没改善时，闭环提升概率低。

## Phase 3：闭环仿真评估

这是核心评估。

### LIBERO

先用 LIBERO 作为快速 sanity check，但不能只看总体 40 task 平均，因为 release baseline 已经很强。

建议三组：

```text
LIBERO full:
  libero_spatial, libero_object, libero_goal, libero_10
  40 tasks x 50 trials

LIBERO hard / long-horizon subset:
  使用 docs/plan/2026-05-14-libero-long-horizon-eval-plan.md 里的 task set
  每 task 100 trials

LIBERO low-step stress:
  num_inference_steps = 1, 2, 4, 10
```

主指标：

```text
success rate overall
success rate per suite
success rate per task
95% confidence interval
```

附加指标：

```text
mean episode length until success/failure
failure mode manual tags
policy latency per replan
GPU memory
```

### RoboTwin

RoboTwin 更适合看 motion token 是否改善困难任务和低步数推理。

建议两级：

```text
quick:
  8 tasks x 20 episodes
  用于快速筛选

formal:
  full task set
  clean + randomized
  100 episodes per task
```

主指标：

```text
clean success rate
randomized success rate
per-task success rate
release_1 / release_4 / release_10 对齐曲线
```

### 闭环对照组

至少跑：

```text
B0 FastWAM release_1
B1 FastWAM release_10
B2 FastWAM fine-tune without motion
B3 FastWAM + zero motion tokens
B4 FastWAM + shuffled motion tokens
B5 FastWAM + predicted motion context
B6 FastWAM + oracle future motion tokens 仅诊断
```

如果做多相机：

```text
B7 external-only motion
B8 wrist-only motion
B9 external + wrist gated residual
B10 external + wrist naive concat
```

### 判定标准

一个结果值得继续投入，需要同时满足：

```text
predicted motion context > no-motion fine-tune
predicted motion context > zero/shuffled motion
提升不仅来自单个 task
latency 增加可接受
低步数推理不明显恶化
```

LIBERO 上建议门槛：

```text
full average 提升 >= 1.0 point，或 hard subset 提升 >= 3.0 points
```

RoboTwin quick 上建议门槛：

```text
8-task quick mean 提升 >= 5.0 points，且至少 4/8 tasks 不下降
```

正式 RoboTwin 上建议门槛：

```text
clean/randomized mean 都不低于 baseline
randomized 或 hard tasks 有显著提升
```

## Phase 4：鲁棒性评估

ZipMotion 的价值不应该只体现在 dataset 内成功率，还应体现在观测扰动和泛化上。

### 相机鲁棒性

```text
camera dropout
wrist camera unavailable
external camera crop/shift
camera pose perturbation
image brightness/noise perturbation
```

指标：

```text
success rate drop = SR_clean - SR_perturbed
```

目标：

```text
ZipMotion 方案的 success drop 小于 FastWAM baseline
```

### 语言 / 任务泛化

```text
seen instruction
paraphrased instruction
unseen object combination
unseen initial object pose
```

### 动作频率 / replan 鲁棒性

```text
replan_steps = 8, 12, 24
control frequency perturbation
```

看 motion token 是否让策略更依赖未来 motion 估计，还是更脆弱。

## Phase 5：真机小规模验证

最终目标是真机泛化，因此 sim 结果只能作为前置筛选。

建议最小真机 suite：

```text
5 tasks
20 trials per task
2 camera setups
```

任务类型：

```text
pick/place
push/align
drawer/open-close
button/switch
occlusion/contact-rich task
```

对照：

```text
FastWAM baseline
FastWAM + predicted motion context
FastWAM + motion disabled at test time
```

指标：

```text
success rate
partial success rate
human intervention count
time to complete
failure taxonomy
latency
```

真机通过门槛：

```text
平均 success rate 提升
至少 3/5 tasks 不下降
延迟不超过控制预算
失败模式没有明显新增安全风险
```

## 统计报告方式

### 置信区间

Success rate 是二项分布，报告时给 95% CI。

简单估计：

```text
SE = sqrt(p * (1-p) / n)
95% CI ≈ p ± 1.96 * SE
```

对于 50 trials：

```text
p=0.90 时 CI 约 ±8.3 points
```

所以单 task 50 trials 波动很大，不要过度解读单 task。

对于 40 tasks x 50 trials = 2000 trials：

```text
p=0.95 时 CI 约 ±1.0 point
```

overall 更可靠，但可能掩盖 hard task 收益。

### 必须报告的表

```text
Table 1: overall success rate
Table 2: per-suite success rate
Table 3: per-task delta vs baseline
Table 4: ablation: predicted / zero / shuffled / oracle motion
Table 5: latency and memory
Table 6: robustness perturbation
```

### 必须画的图

```text
step curve: num_inference_steps 1/2/4/10 vs success rate
per-task delta bar chart
motion quality vs policy success scatter
latency-success tradeoff
```

## 实验执行顺序

推荐不要一上来 full eval，按下面顺序止损：

```text
1. Information leakage audit
2. ZipMotion predictor metric on held-out data
3. open-loop action prediction with zero/shuffled/predicted motion
4. LIBERO 5-task smoke, 20 trials each
5. RoboTwin quick 8-task clean20
6. LIBERO full 40-task 50 trials
7. LIBERO hard subset 100 trials
8. RoboTwin formal clean/randomized
9. real-robot small suite
```

每一步的 go/no-go：

```text
Phase 1 fail: 不接 FastWAM，先修 motion predictor
Phase 2 fail: 修 adapter / fusion，不跑闭环 full
Phase 3 quick fail: 不跑 formal，先做 ablation
Phase 3 pass: 扩到 full 和 robustness
```

## 推荐的第一轮最小实验

第一轮目标不是发结论，而是判断 ZipMotion 接入是否值得继续。

### 训练组

```text
M0: FastWAM fine-tune without motion
M1: FastWAM + predicted motion context
M2: FastWAM + zero motion context
M3: FastWAM + shuffled motion context
M4: FastWAM + oracle motion context 仅诊断
```

### 评估

```text
LIBERO selected 5 tasks x 20 trials
RoboTwin quick 8 tasks x 20 episodes
open-loop action validation
```

### 预期判定

如果结果是：

```text
M1 > M0, M2, M3
M4 明显 > M1
```

说明：

```text
motion 通路有效，但 predictor 仍有提升空间
```

如果结果是：

```text
M4 > M0, 但 M1 ≈ M0
```

说明：

```text
motion 条件理论有用，但当前 predictor 不够准
```

如果结果是：

```text
M4 ≈ M0
```

说明：

```text
FastWAM 没用上 motion token，或任务不依赖这类 motion 信息
```

如果结果是：

```text
M1 < M0 且 M2/M3 也下降
```

说明：

```text
adapter 引入了干扰，需要 gate 初始化、motion dropout 或更小学习率
```
