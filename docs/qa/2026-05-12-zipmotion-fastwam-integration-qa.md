# 问答归档：ZipMotion / ZipMo 与 FastWAM 集成分析

日期：2026-05-12

## Q1：ZipMo 的基础预训练用了哪些数据？

从 `long-term-motion` 仓库和项目页表述看，ZipMo 的基础 motion embedding / VAE 主要来自 open-domain / internet videos 上预提取的 tracker trajectories。

关键依据：

- README 说 ZipMo 的 long-term motion embedding 是从 large-scale tracker trajectories 学到的。
- Training 章节要求先 collect videos、shard 成 `webdataset`，并 pre-extract tracker trajectories。
- `scripts/train.py` 默认训练数据路径名是 `koala-tapnext-subset`。
- 数据 loader 读的是 `video.mp4`、`tracks_yx.npy`、`visibility.npy`、可选 `certainty.npy`。

因此，基础预训练不应理解为用大量 LIBERO 仿真数据训练出来。

## Q2：LIBERO 在 ZipMo 里是不是完全没有参与训练？

不能这么说。

更准确的拆分是：

1. 通用 motion embedding / VAE 预训练：主要是互联网/open-domain 视频轨迹数据。
2. LIBERO 专用 planner / policy head：应该用 LIBERO 任务数据做过训练或适配。
3. 开源仓库当前主要提供 LIBERO 的 evaluation / rollout 入口和已发布权重。

原文说：在 LIBERO 中，模型根据 task description 和 start frame 预测物体运动，再训练一个 small policy head，把 generated motions 映射到 7D robot actions。

代码侧也能看出 policy head 是有监督训练，不是只靠 rollout：

- `PolicyHead.forward(...)` 输入包含 `actions` 真值。
- 先用冻结的 track predictor 生成 motion latent。
- 再预测 `actions_pred`。
- loss 是 `(actions_pred - actions) ** 2`。

rollout 是评测，不是训练动作的唯一来源。

## Q3：我们为什么不建议在核心表示里放 dataset embedding？

我们的目标是真机泛化，而不是只在 Bridge、DROID 或 LIBERO 上过拟合。

如果在核心模型里加入：

```text
E_DROID
E_Bridge
E_LIBERO
```

模型容易学成数据集风格条件化，例如相机布局、控制频率、背景分布、动作 convention。部署真机时没有可靠的 “dataset id”，这种 shortcut 可能伤害泛化。

更推荐放入可部署时也成立的物理/观测信息：

```text
E_wrist
E_external
E_overhead
camera intrinsics / extrinsics
control frequency
action frame convention
robot / gripper physical descriptor
proprio schema mask
```

per-dataset adapter 可以存在，但建议只放在输入输出边界，用来处理 action space、proprio 维度、相机格式、控制频率差异，不要进入核心 motion reasoning。

## Q4：camera role embedding 是什么，怎么学习出来？

camera role embedding 是一组可学习参数，用来告诉模型某些 tokens 来自哪个相机角色。

示例：

```python
external_tokens = external_tokens + E_external
wrist_tokens = wrist_tokens + E_wrist
tokens = concat([external_tokens, wrist_tokens])
```

`E_external` 和 `E_wrist` 不是手工规则，而是和模型权重一起通过反向传播学习。训练目标可以是 track reconstruction、future motion prediction、action prediction 或它们的组合。

它解决的是“这些 tokens 来自哪个视角”的身份问题，不等于决定信息怎么融合。

## Q5：wrist 相机变化快、难利用，是否应该 external 主干、wrist 辅助？

是的，这是当前更合理的方向。

external camera 通常更适合作为主干：

- 视野大。
- 相机稳定。
- 更容易看到目标物、桌面布局、任务进展。
- 对 long-horizon planning 更可靠。

wrist camera 的价值在于：

- 末端附近细节。
- 接触、抓取、遮挡恢复。
- 但运动快、模糊多、视野窄。

所以当前建议是：

```text
external 作为主 motion latent
wrist 作为 gated residual / auxiliary correction
```

但这不是像素残差，也不是把 wrist 图像直接拼到 external 图像上。

## Q6：这个 residual 设计是否考虑了 ZipMotion 是点追踪模型？

需要特别强调：ZipMotion 的核心对象不是普通图像 latent，而是 start frame + 2D point tracks。

因此更贴合 ZipMotion 的形式应写成：

```python
z_ext = ZipMoVAE.encode(
    tracks_ext,
    start_ext_frame,
)

z_wrist = ZipMoVAE.encode(
    tracks_wrist,
    start_wrist_frame,
)

delta = CrossAttention(
    query=z_ext,
    key_value=z_wrist,
)

gate = sigmoid(MLP(z_ext, z_wrist, proprio))

z_motion = z_ext + gate * delta
```

这里的 residual 发生在 motion latent / token 层，不发生在像素层，也不发生在 2D track 坐标层。

关键原因：external tracks 和 wrist tracks 不在同一个 image plane，通常没有一一对应的 point identity，直接相加没有几何意义。

## Q7：FastWAM 当前训练和推理机制是什么？

FastWAM 当前是 Wan 视频 DiT + ActionDiT + MoT 的结构。

训练时：

```text
多相机图像 -> 拼成一个视频画布 -> Wan VAE -> video latents
语言指令 -> T5 context
proprio -> linear -> append 到 context
action chunk -> 连续动作序列
```

然后对 video latent 和 action 同时做 continuous flow-matching：

```text
x_t = (1 - sigma) * x + sigma * noise
target = noise - x
```

loss 是：

```text
loss = lambda_video * video_velocity_mse
     + lambda_action * action_velocity_mse
```

推理动作时，FastWAM 不生成完整未来视频。它只编码当前第一帧得到 video tokens，prefill KV cache，然后从随机 action noise 开始迭代去噪 action chunk。

MoT attention mask 里，action tokens 可以看 action 自身和第一帧 video tokens。

## Q8：MoT expert 是什么，MoT 全称是什么？

在当前 FastWAM 代码里，`MoT` 可以理解为 Mixture of Transformers。

它包含至少两个 expert：

```python
mixtures = {
    "video": video_expert,
    "action": action_expert,
}
```

expert 是处理特定 token 类型的 Transformer 分支：

- `video_expert` 处理 video latent tokens。
- `action_expert` 处理 action tokens。

MoT 的重点是：不同 expert 有各自的投影、MLP、调制和输出头，但 self-attention 时可以把 video/action tokens 混合起来做 attention。

## Q9：把 ZipMotion tokens 当额外 context 给 FastWAM cross-attend，具体是什么意思？

FastWAM 已经有 `context/context_mask` 通路，语言 tokens 和 proprio token 会作为 cross-attention 的 key/value 被 action/video expert 读取。

如果 ZipMotion predictor 输出：

```text
Z_motion ∈ R^{B × L_m × d_z}
```

则先投影到 FastWAM context 维度：

```text
C_motion = Z_motion W_m + b_m
C_motion ∈ R^{B × L_m × 4096}
```

再拼到原 context 后面：

```text
C' = concat(C_text, C_prop, C_motion)
M' = concat(M_text, M_prop, M_motion)
```

ActionDiT 的 cross-attention 从：

```text
Attn(X_action, C)
```

变成：

```text
Attn(X_action, C')
```

通俗地说，就是给 FastWAM 增加一种提示：

```text
语言提示：要做什么
proprio 提示：当前机械臂状态
ZipMotion 提示：未来物体/场景可能怎么动
```

## Q10：接入 ZipMotion 时最大的原则是什么？

不能把未来真实 tracks 当作 policy 输入。

未来帧跑 tracker 得到的 tracks 可以作为训练监督目标，但不能作为推理输入。否则会产生信息泄漏。

正确方式：

```text
训练输入：当前/历史观测 -> ZipMotion predictor -> predicted motion tokens
训练监督：未来 tracks / future motion latent / action 可用于 loss target
推理输入：当前/历史观测 -> ZipMotion predictor -> predicted motion tokens
```

## Q11：除了 context token adapter，还有哪些适配方式？

主要候选：

1. Context Token Adapter
   把 ZipMotion tokens 投影后拼进 FastWAM context。改动最小，适合第一版。

2. Gated Cross-Attention Injection
   在 action expert 若干层加入：

   ```text
   h_action = h_action + sigmoid(g) * CrossAttn(h_action, z_motion)
   ```

   gate 可初始化为 0，降低破坏原模型的风险。这个思路接近 LaMP / PointVLA 等工作。

3. Cascaded Motion-Then-Action
   先预测未来物体/点轨迹运动，再用该运动指导动作生成。思路接近 Motion Before Action。

4. Motion Expert 加入 MoT
   把 ZipMotion 做成第三个 expert：

   ```python
   mixtures = {
       "video": video_expert,
       "action": action_expert,
       "motion": motion_expert,
   }
   ```

   结构最统一，但工程风险最大，不建议第一版做。

5. Auxiliary Motion Loss
   不把 ZipMotion tokens 输入 FastWAM，只让 FastWAM hidden/action tokens 预测 ZipMotion latent，作为辅助监督。

## Q12：当前最稳妥路线是什么？

按风险从低到高：

1. 先训练/微调机器人版 ZipMotion，让它从当前/历史多相机观测预测 future motion latent。
2. 用 frozen ZipMotion predictor 生成 motion tokens。
3. 把 motion tokens 作为额外 context 拼给 FastWAM。
4. 继续训练 FastWAM action flow loss，先验证 action loss 和 rollout 是否改善。
5. 如果有效，再做 gated cross-attention injection。
6. 只有前两步确认有效后，再考虑把 ZipMotion 做成第三个 MoT expert。

关键结论：

```text
ZipMotion 可以增强 FastWAM，但必须作为 causal future-motion prior 或 auxiliary target 接入；
不能把未来真实点轨迹当作 action policy 输入。
```

## 参考工作

- DROID: https://droid-dataset.github.io/
- BridgeData V2: https://bridgedata-v2.github.io/
- Octo: https://octo-models.github.io/
- RDT: https://rdt-robotics.github.io/rdt-robotics/
- PointVLA: https://pointvla.github.io/
- LaMP: https://papers.cool/arxiv/2603.25399
