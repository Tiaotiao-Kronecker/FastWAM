# ZipMotion / ZipMo 与 FastWAM 集成计划

日期：2026-05-12

## 目标

增强 ZipMotion / ZipMo motion embedding 对下游机械臂策略的支持，并将其接入 FastWAM / VLA 类策略，最终目标是真机泛化，而不是只在 Bridge、DROID、LIBERO 或 RoboTwin 上得到数据集内收益。

核心方向：

```text
多相机机器人视频
  -> causal future-motion predictor
  -> motion tokens / motion latent
  -> FastWAM action generator 的额外条件或辅助监督
```

## 当前判断

### ZipMotion 侧

ZipMo 的基础 motion embedding / VAE 是点追踪形式：

```text
start frame + 2D point tracks -> motion latent
motion latent + query points + start frame -> future tracks
```

因此，多相机融合不应理解成普通图像特征融合，而应理解成 per-view point-track latent 融合。

推荐基础结构：

```text
external tracks + external start frame
  -> ZipMo encoder
  -> z_ext

wrist tracks + wrist start frame
  -> ZipMo encoder
  -> z_wrist

z_motion = z_ext + gate * CrossAttn(z_ext <- z_wrist, proprio)
```

含义：

- external 是主 motion latent。
- wrist 是 gated residual / auxiliary correction。
- residual 发生在 latent/token 层，不发生在像素层，也不发生在 2D track 坐标层。

### FastWAM 侧

FastWAM 当前训练是 video/action continuous flow-matching：

```text
x_t = (1 - sigma) * x + sigma * noise
target = noise - x
```

训练 loss：

```text
loss = lambda_video * video_velocity_mse
     + lambda_action * action_velocity_mse
```

推理动作时只用第一帧 video tokens prefill cache，然后迭代去噪 action chunk。

当前多相机处理是像素拼接，例如 RobotWin 把 top camera 和左右 wrist camera 拼成一张视频画布。这对 ZipMotion 接入不是理想表示，因为 ZipMotion 更适合 per-view point-track latent。

## 关键原则

1. 未来真实 tracks 只能作为训练监督，不能作为 FastWAM policy 输入。
2. 核心 motion latent 应尽量 dataset-agnostic。
3. 不建议在核心模型里使用 `E_DROID`、`E_Bridge`、`E_LIBERO` 这类 dataset id embedding。
4. 推荐使用 camera role / physical descriptor：

   ```text
   E_external
   E_wrist
   E_overhead
   camera intrinsics / extrinsics
   control frequency
   robot / gripper physical descriptor
   proprio schema mask
   ```

5. per-dataset adapter 只作为边界层，用于处理 action space、proprio schema、相机格式和控制频率差异。
6. 第一版应尽量不破坏 FastWAM 原有 MoT 结构。

## 推荐方案 A：Context Token Adapter

这是第一优先级方案。

### 形式

ZipMotion predictor 输出：

```text
Z_motion ∈ R^{B × L_m × d_z}
```

投影到 FastWAM context 维度：

```text
C_motion = Z_motion W_m + b_m
C_motion ∈ R^{B × L_m × 4096}
```

拼接到原 context：

```text
C' = concat(C_text, C_prop, C_motion)
M' = concat(M_text, M_prop, M_motion)
```

FastWAM action expert 原本 cross-attend：

```text
Attn(X_action, C)
```

现在变成：

```text
Attn(X_action, C')
```

### 工程改动

候选新增模块：

```text
src/fastwam/models/wan22/zipmotion_context.py
```

候选改动点：

- `FastWAM.__init__`：增加 optional `motion_context_encoder` / `motion_proj`。
- `FastWAM.build_inputs`：从 sample 中读取或计算 `motion_tokens`。
- `FastWAM.infer_action`：支持传入 `motion_tokens` 或内部调用 frozen predictor。
- configs：新增 `model.motion_context` 配置开关。

第一版可先离线预计算 `motion_tokens`，降低工程复杂度。后续再把 predictor 放进在线推理路径。

### Loss

第一版保持 FastWAM 原 action flow loss 不变：

```text
L = L_action_flow
```

可选增加：

```text
L = L_action_flow + lambda_motion * L_motion_aux
```

但不要一开始引入太多 loss。

### 优点

- 改动小。
- 能复用 FastWAM 已有 context cross-attention。
- 不需要修改 MoT attention mask 和 KV cache。
- 适合快速 ablation。

### 风险

- motion tokens 只通过 context 被读取，利用强度可能不足。
- 如果 motion tokens 质量差，可能被模型忽略。
- context 序列变长会增加 attention 开销。

## 推荐方案 B：Gated Cross-Attention Injection

这是第二阶段正式方案。

### 形式

在 action expert 若干层注入：

```text
h_action = h_action + sigmoid(g) * CrossAttn(h_action, z_motion)
```

gate 建议初始化为接近 0，使初始模型近似原 FastWAM。

### 工程改动

候选新增模块：

```text
MotionCrossAttentionAdapter
```

插入位置：

- action expert 后若干层。
- 或 MoT post-block 之后、action token 更新之前。

### 优点

- 比 context token adapter 更直接。
- gate 可以控制 ZipMotion 干预强度。
- 更接近 LaMP / PointVLA 的 adapter 思路。

### 风险

- 需要改 action expert / MoT block。
- 需要处理训练稳定性。
- KV cache 推理路径也要同步适配。

## 暂不优先：Motion Expert 加入 MoT

形式：

```python
mixtures = {
    "video": video_expert,
    "action": action_expert,
    "motion": motion_expert,
}
```

不建议第一版做。

原因：

- 需要 motion expert 与 video/action expert 对齐 layer 数、head 数、hidden 维度和 RoPE。
- 需要重新设计 attention mask。
- 需要适配 KV cache。
- 会显著增加调试复杂度。

只有 Context Token Adapter 和 Gated Cross-Attention 都确认有效后，再考虑这个方向。

## 训练路线

### Phase 0：数据与轨迹预处理

目标：为 Bridge / DROID / 自采真机数据构建 per-view tracks。

输入：

```text
external camera video
wrist camera video
optional overhead / side camera video
proprio
action
language / task
camera metadata
```

处理：

1. 每个 camera view 独立跑 tracker。
2. 保存：

   ```text
   tracks_yx.npy
   visibility.npy
   certainty.npy
   video.mp4 或 frames
   camera_role
   optional intrinsics/extrinsics
   ```

3. 对不同数据集统一 camera role 命名：

   ```text
   external
   wrist
   overhead
   left_static
   right_static
   ```

4. 训练时使用 view dropout / camera dropout，防止模型依赖固定相机组合。

### Phase 1：机器人版 ZipMotion 预训练 / 微调

目标：让 ZipMotion 从机器人多相机视频中学习 action-relevant future motion。

模型：

```text
z_ext = ZipMoVAE.encode(tracks_ext, start_ext_frame)
z_wrist = ZipMoVAE.encode(tracks_wrist, start_wrist_frame)
z_motion = z_ext + gate * CrossAttn(z_ext <- z_wrist, proprio)
```

训练目标：

```text
L_track_ext = reconstruct external future tracks
L_track_wrist = reconstruct wrist future tracks
L_action_align = predict EEF/action chunk or align with action representation
```

总 loss：

```text
L_zipmotion =
  lambda_ext * L_track_ext
  + lambda_wrist * L_track_wrist
  + lambda_action * L_action_align
```

注意：

- future tracks 可以作为 loss target。
- 推理输入只能用当前/历史帧可预测出的 motion tokens。

### Phase 2：Context Token Adapter 接入 FastWAM

目标：最小侵入式验证 ZipMotion 是否能提升 FastWAM action。

训练输入：

```text
当前观测
语言
proprio
ZipMotion predicted motion tokens
action ground truth
```

FastWAM 结构：

```text
context = concat(text_context, proprio_context, motion_context)
```

训练目标：

```text
L = L_action_flow
```

可选：

```text
L = L_action_flow + lambda_motion * L_motion_aux
```

第一版建议：

- 冻结 ZipMotion predictor。
- 训练 `motion_proj` + FastWAM action 相关参数。
- 先不训练 video expert。
- 保留原 FastWAM action flow loss。

### Phase 3：Gated Cross-Attention Adapter

如果 Phase 2 有收益，再引入 gated injection。

结构：

```text
h_action = h_action + gate * CrossAttn(h_action, z_motion)
```

训练策略：

- gate 初始化为 0 或很小。
- 先只开放 adapter 和 action expert 小学习率。
- 再考虑解冻更多 MoT/action 参数。

### Phase 4：真机泛化验证

必须做跨数据集和真机验证，而不是只看 Bridge/DROID 内部 validation。

建议验证：

```text
train: Bridge + DROID
eval: held-out DROID scenes
eval: held-out Bridge tasks
eval: LIBERO/RoboTwin sim sanity check
eval: 少量自采真机任务
```

关键 ablation：

```text
FastWAM baseline
FastWAM + external-only ZipMotion
FastWAM + wrist-only ZipMotion
FastWAM + external主干 + wrist residual
FastWAM + context token adapter
FastWAM + gated cross-attention
```

指标：

- action MSE / endpoint error
- rollout success rate
- replan latency
- view dropout robustness
- unseen camera placement robustness
- small real-world finetune sample efficiency

## 重点风险

### 信息泄漏

最大风险是误把未来真实 tracks 输入 action policy。

硬性规则：

```text
future tracks 只能作为 target，不允许作为 policy input。
```

### 表示错位

FastWAM 当前是拼图视频 latent，ZipMotion 是 per-view point-track latent。直接替换 video input 不是好第一步。

推荐：

```text
先作为 context / adapter 接入
后续再考虑替换多相机像素拼接
```

### wrist 视角噪声

wrist 视角变化快，不应强制平权。

推荐：

```text
external 主干
wrist gated residual
view dropout
gate 正则
```

### dataset shortcut

避免核心模型记住数据集 ID。

推荐：

```text
不用核心 dataset embedding
使用 camera role / physical descriptor
dataset adapter 只放边界层
```

## 最小下一步

1. 为 FastWAM 设计 `motion_context` 配置，不默认启用。
2. 做一个离线 `motion_tokens` mock 数据路径，先验证 context 拼接和训练路径。
3. 新增 `motion_proj: Linear(d_z, 4096)`。
4. 在 `FastWAM.build_inputs` 里 append motion context。
5. 跑一个小 batch forward/backward smoke。
6. 再接真实 ZipMotion predictor 或离线预计算 tokens。

## 参考工作

- DROID: https://droid-dataset.github.io/
- BridgeData V2: https://bridgedata-v2.github.io/
- Octo: https://octo-models.github.io/
- RDT: https://rdt-robotics.github.io/rdt-robotics/
- PointVLA: https://pointvla.github.io/
- LaMP: https://papers.cool/arxiv/2603.25399
