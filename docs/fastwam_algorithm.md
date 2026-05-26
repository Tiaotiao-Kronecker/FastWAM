<div class="layout">

## FastWAM 算法页

[1. 总览](#overview) [2. 符号与张量](#symbols) [3. 数据处理](#data) [4. 条件 context](#context) [5. Flow matching](#flow) [6. Video/Action Expert](#experts) [7. MoT 与 Cross Attention](#attention) [8. 训练 Loss](#loss) [9. 推理动作生成](#inference) [10. 接入 ZipMotion 的位置](#zipmotion) [11. 源码索引](#source-index)

<div role="main">

# FastWAM 算法过程、数学公式与源码映射

本页把当前仓库里的 FastWAM 实现拆成完整数据流：多相机视频和机器人状态如何进入模型， video/action 两个专家如何通过 MoT 交互，cross attention 如何读取文本和 proprio 条件， 训练时如何构造 flow-matching 目标，推理时如何从当前帧生成动作序列。

<div class="meta">

<span class="tag">源码根目录：src/fastwam</span> <span class="tag">模型：Wan2.2 Video Expert + ActionDiT + MoT</span> <span class="tag">生成日期：2026-05-13</span>

</div>

<div id="overview" class="section">

## 1\. 总览

FastWAM 在这个仓库里可以理解为一个双专家 flow-matching 模型： **Video Expert** 负责在 VAE latent 空间预测视频 flow， **Action Expert** 负责在 action 空间预测动作 flow， **MoT** 负责把 video tokens 和 action tokens 放进同一个 attention 通道里交互。

关键点是：图像本身不是 cross attention 的 context。当前实现中，图像先经 VAE 和 VideoDiT 变成 video tokens； 文本指令和 proprio 被组织成 context tokens，被 video/action 两个专家通过 cross attention 读取。

<div class="diagram-wrap" aria-label="FastWAM 数据流示意图">

多相机视频 \[Ncam,T,C,H,W\] 拼接/resize/normalize video: \[B,3,T,H,W\] Wan VAE z0: \[B,48,F,h,w\] Video noising z\_sigma, target Video Expert video tokens 动作序列 a0: \[B,Ta,Da\] Action noising a\_sigma, target Action Expert action tokens MoT 混合注意力 concat video/action masked self-attn split back to experts Cross Attention Q: expert tokens K/V: context tokens text + proprio 任务文本 instruction T5/Wan context cache Ctxt: \[B,L,4096\] Proprio token \[B,1,4096\] Heads predict video/action flow Loss / Inference MSE 或 denoise step

<div class="legend">

<span><span class="dot blue"></span>视频路径</span> <span><span class="dot red"></span>动作路径</span> <span><span class="dot green"></span>条件 context 路径</span> <span><span class="dot amber"></span>专家交互路径</span>

</div>

</div>

</div>

<div id="symbols" class="section">

## 2\. 符号与张量

| 符号               | 含义                        | 典型形状                                                  | 来源                             |
| ---------------- | ------------------------- | ----------------------------------------------------- | ------------------------------ |
| \\(B\\)          | batch size                | 训练时可大于 1，动作推理时通常为 1                                   | DataLoader / infer\_action     |
| \\(V\\)          | 归一化视频                     | \\(\[B,3,T,H,W\]\\)                                   | dataset 输出 `sample["video"]`   |
| \\(Z\_0\\)       | VAE latent video          | \\(\[B,48,F,h,w\]\\)                                  | Wan VAE encode                 |
| \\(A\_0\\)       | 真实动作序列                    | \\(\[B,T\_a,d\_a\]\\)                                 | dataset 输出 `sample["action"]`  |
| \\(C\\)          | 条件 context，含文本和可选 proprio | \\(\[B,L,4096\]\\) 或 \\(\[B,L+1,4096\]\\)             | T5/Wan cache + proprio encoder |
| \\(X\_v, X\_a\\) | Video/Action expert token | \\(X\_v:\[B,S\_v,3072\]\\)，\\(X\_a:\[B,T\_a,1024\]\\) | pre\_dit                       |
| \\(H, d\_h\\)    | attention head 数和每头维度     | \\(H=24,d\_h=128\\)                                   | `configs/model/fastwam.yaml`   |

<div class="callout">

注意：Action Expert 的 residual hidden dim 是 1024，但 attention 内部维度是 \\(H d\_h=24\\times128=3072\\)。 这是通过 `Linear(1024,3072)` 的可学习投影完成的，attention 后再用 `Linear(3072,1024)` 投回主干维度。

</div>

### 公式阅读约定

| 记号                                             | 详细含义                                                                                                                   | 代码变量或配置                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| \\(T\\), \\(T\_v\\), \\(T\_a\\)                | \\(T\\) 是原始观测帧数；\\(T\_v\\) 是抽样后送入模型的视频帧数；\\(T\_a\\) 是动作 horizon。训练要求 action horizon 能被视频 transition 数 \\(T\_v-1\\) 整除。 | `num_frames`, `T_video`, `action_horizon`                                 |
| \\(H\_o,W\_o\\)                                | 拼接、resize、crop 后送入 VAE 的图像高度和宽度。模型要求二者都是 16 的倍数。                                                                       | `height`, `width`                                                         |
| \\(F,h,w\\)                                    | VAE latent 的时间、高度、宽度维度；它们不是原始像素维度。VideoDiT patchify 后得到 \\(F',h',w'\\)。                                                | `input_latents.shape[2:]`                                                 |
| \\(D\_v,D\_a,D\_c\\)                           | Video Expert hidden dim、Action Expert hidden dim、原始 context dim。当前配置分别是 3072、1024、4096。                                | `video_dit_config.hidden_dim`, `action_dit_config.hidden_dim`, `text_dim` |
| \\(S\_v,S\_a,L\_c\\)                           | video token 数、action token 数、context token 数。通常 \\(S\_a=T\_a\\)，\\(L\_c=L\\) 或 \\(L+1\\) 或 \\(L+1+L\_m\\)。             | `video_tokens.shape[1]`, `action_tokens.shape[1]`, `context.shape[1]`     |
| \\(W,b\\)                                      | 公式中的 \\(W\\) 和 \\(b\\) 都是可学习线性层参数，不是手工定义的常量。                                                                           | `nn.Linear(...)`                                                          |
| \\(\\sigma,t,u,s\\)                            | \\(u\\) 是均匀随机数；\\(s\\) 是 scheduler shift；\\(\\sigma\\) 是连续噪声强度；\\(t\\) 是缩放到训练步数区间的 timestep。                           | `u`, `shift`, `sigma`, `timestep`                                         |
| \\(\\epsilon\\), \\(v^\\star\\), \\(\\hat v\\) | \\(\\epsilon\\) 是高斯噪声；\\(v^\\star=\\epsilon-X\_0\\) 是训练目标速度；\\(\\hat v\\) 是模型预测速度。                                     | `noise_*`, `target_*`, `pred_*`                                           |

### 源码变量名对照

  - `input_latents`
    公式里的 \\(Z\_0\\)，即 clean video latent。
  - `latents`
    公式里的 \\(Z\_\\sigma\\)，即加噪 video latent。
  - `action`
    公式里的 \\(A\_0\\)，即 clean action sequence。
  - `noisy_action`
    公式里的 \\(A\_\\sigma\\)，即加噪 action sequence。
  - `context`
    公式里的 \\(C\\)，即文本/proprio/motion 等条件 token 的集合。
  - `tokens`, `x_tokens`
    公式里的 \\(X\_a\\) 或 \\(X\_v\\)，具体含义取决于当前 expert。
  - `context_mask`, `attention_mask`
    布尔 mask；在 PyTorch `scaled_dot_product_attention` 里，`True` 表示允许 attention。

</div>

<div id="data" class="section">

## 3\. 数据处理

训练样本首先由 LeRobot 风格数据集读出多相机图像、动作、状态和任务文本。 当前 FastWAM 的多相机进入方式是**像素级画布拼接**，不是多相机 token cross attention。

<div class="formula-card">

<div class="formula-head">

**F1. 多相机视频拼接** [robot\_video\_dataset.py:142-197](../src/fastwam/datasets/lerobot/robot_video_dataset.py)

</div>

<div class="formula-body">

\\\[ I \\in \\mathbb{R}^{N\_c \\times T \\times 3 \\times H \\times W} \\quad\\Longrightarrow\\quad V \\in \[-1,1\]^{3 \\times T\_v \\times H\_o \\times W\_o} \\\]

对 RobotWin 三相机，top 相机被 resize 到 \\(256\\times320\\)，左右 wrist 相机被 resize 到 \\(128\\times160\\)， 左右 wrist 横向拼接后再和 top 纵向拼接，得到 \\(384\\times320\\) 的画布。

### 参数说明

  - \\(I\\)
    processor 输出的多相机图像张量，对应 `sample["pixel_values"]`。
  - \\(N\_c\\)
    相机数量；RobotWin 模式要求正好 3 个相机。
  - \\(T\\), \\(T\_v\\)
    \\(T\\) 是原始观测窗口帧数，\\(T\_v\\) 是 `video_sample_indices` 抽样后的帧数。
  - \\(H\_o,W\_o\\)
    最终送入 VAE 的画布尺寸，由拼接、resize、crop 后决定。
  - \\(\[-1,1\]\\)
    图像归一化范围，来自 `normalize_transform`。

<!-- end list -->

    video = sample["pixel_values"]  # [num_cameras, T, C, H, W]
    cam_top = resize(video[0], size=[256, 320])
    cam_left = resize(video[1], size=[128, 160])
    cam_right = resize(video[2], size=[128, 160])
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    video = torch.cat([cam_top, bottom], dim=-2)
    video = self.normalize_transform(video)
    video = video.permute(1, 0, 2, 3)  # [C, T, H, W]

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F2. 动作和 proprio 对齐** [robot\_video\_dataset.py:199-203](../src/fastwam/datasets/lerobot/robot_video_dataset.py)

</div>

<div class="formula-body">

\\\[ A\_0 = \\{a\_0,\\ldots,a\_{T\_a-1}\\},\\qquad P = \\{p\_0,\\ldots,p\_{T\_a-1}\\} \\\]

数据注释里写明 action 从 \\(t\_0\\) 开始但少最后一帧，proprio 原始长度和视频帧对齐。 当前代码使用 `proprio[:-1]` 让 proprio 与 action 对齐。

### 参数说明

  - \\(A\_0\\)
    clean action sequence，是 flow matching 的动作训练目标基础。
  - \\(a\_\\tau\\)
    第 \\(\\tau\\) 个动作向量，维度是 \\(d\_a=\\)`action_dim`。
  - \\(P\\)
    与动作时间步对齐后的 proprio 序列。
  - \\(p\_\\tau\\)
    第 \\(\\tau\\) 个机器人状态向量，维度是 `proprio_dim`。
  - \\(T\_a\\)
    动作 horizon，也就是 `action.shape[1]`。

<!-- end list -->

    action = sample["action"]         # [T-1, action_dim]
    proprio = sample["proprio"][:-1]  # [T-1, state_dim]

</div>

</div>

</div>

<div id="context" class="section">

## 4\. 条件 Context

FastWAM 的 cross attention 读取的 context 主要来自两类信息：任务文本 embedding 和可选 proprio token。 文本 embedding 当前通常是离线预计算的 Wan/T5 context cache。

<div class="formula-card">

<div class="formula-head">

**F3. 文本 context cache** [robot\_video\_dataset.py:218-268](../src/fastwam/datasets/lerobot/robot_video_dataset.py)

</div>

<div class="formula-body">

\\\[ C\_{\\text{text}} = E\_{\\text{text}}(\\text{instruction}) \\in \\mathbb{R}^{L \\times 4096} \\\]

dataset 根据 prompt 的 hash 读取预计算文件，返回 `context` 和 `context_mask`。 之后把无效 token 置零，并把 mask 设成全 1，以保持和 Wan2.2 行为一致。

### 参数说明

  - \\(C\_{\\text{text}}\\)
    文本编码器输出的 token 序列；每个 token 是 4096 维。
  - \\(E\_{\\text{text}}\\)
    Wan/T5 文本编码器；训练时通常不在线运行，而是读取 cache。
  - \\(L\\)
    文本 token 长度，当前配置常用 `context_len=128`。
  - `context_mask`
    标记哪些 context token 有效；当前 dataset 为兼容 Wan 行为把它设成全 True。

<!-- end list -->

    context, context_mask = self._get_cached_text_context(instruction)
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F4. Proprio 作为额外 context token** [fastwam.py:219-240](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ c\_p = W\_p p\_0 + b\_p \\in \\mathbb{R}^{4096} \\\] \\\[ C = \\operatorname{Concat}(C\_{\\text{text}}, c\_p) \\in \\mathbb{R}^{B \\times (L+1) \\times 4096} \\\]

训练时当前实现只取 episode 当前窗口的第一步 proprio：`proprio = proprio[:, 0, :]`。 这意味着 proprio 是当前状态条件，而不是整段 proprio 序列条件。

### 参数说明

  - \\(p\_0\\)
    当前观测窗口第一步 proprio，对应训练代码里的 `proprio[:, 0, :]`。
  - \\(W\_p,b\_p\\)
    `proprio_encoder` 的权重和偏置；代码中是 `nn.Linear(proprio_dim, text_dim)`。
  - \\(c\_p\\)
    proprio 被投影后的单个 context token，形状是 \\(\[B,1,4096\]\\)。
  - \\(C\\)
    拼接后的条件 token；之后会分别进入 Video Expert 和 Action Expert 的 `text_embedding` MLP。

<!-- end list -->

    self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim)
    proprio_token = self.proprio_encoder(proprio.unsqueeze(1))  # [B, 1, D]
    context = torch.cat([context, proprio_token], dim=1)
    context_mask = torch.cat([context_mask, proprio_mask], dim=1)

</div>

</div>

</div>

<div id="flow" class="section">

## 5\. Flow Matching 噪声路径

FastWAM 对 video latent 和 action 都使用同一种 continuous flow-matching scheduler。 训练时从 clean sample 和高斯噪声之间采样一个中间点，模型学习从 clean 指向 noise 的速度场。 推理时从噪声出发，沿着相反方向积分回 clean action 或 video。

<div class="formula-card">

<div class="formula-head">

**F5. shift 后的时间采样** [scheduler\_continuous.py:17-37](../src/fastwam/models/wan22/schedulers/scheduler_continuous.py)

</div>

<div class="formula-body">

\\\[ u \\sim \\mathcal{U}(0,1),\\qquad \\sigma = \\phi(u;s)=\\frac{s u}{1+(s-1)u},\\qquad t = N\_{\\text{steps}}\\sigma \\\]

`shift` 控制训练和推理的噪声时间分布。配置里 video/action 默认 shift 都是 5.0。

### 参数说明

  - \\(u\\)
    从 \\(\[0,1\]\\) 均匀采样的随机数，每个 batch 样本一个。
  - \\(s\\)
    scheduler shift，控制采样更偏向高噪声还是低噪声区域。
  - \\(\\phi(u;s)\\)
    shift 变换函数，把均匀分布的 \\(u\\) 映射为噪声强度 \\(\\sigma\\)。
  - \\(N\_{\\text{steps}}\\)
    训练 timestep 总数，默认 1000。
  - \\(t\\)
    传入模型 time embedding 的 timestep，等于 \\(\\sigma\\) 乘以训练步数。

<!-- end list -->

    u = torch.rand((batch_size,), device=device, dtype=torch.float32)
    sigma = self._phi(u, self.shift)
    timestep = sigma * float(self.num_train_timesteps)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F6. 加噪样本和训练目标** [scheduler\_continuous.py:49-61](../src/fastwam/models/wan22/schedulers/scheduler_continuous.py)

</div>

<div class="formula-body">

\\\[ X\_{\\sigma} = (1-\\sigma)X\_0+\\sigma\\epsilon,\\qquad v^\\star = \\epsilon-X\_0 \\\]

对 video，\\(X\_0=Z\_0\\)。对 action，\\(X\_0=A\_0\\)。 模型输出 \\(\\hat v\_\\theta(X\_\\sigma,t,C)\\)，训练时用 MSE 拟合 \\(v^\\star\\)。

### 参数说明

  - \\(X\_0\\)
    clean sample；可以是 clean video latent \\(Z\_0\\)，也可以是 clean action \\(A\_0\\)。
  - \\(\\epsilon\\)
    与 \\(X\_0\\) 同形状的标准高斯噪声，对应 `torch.randn_like(...)`。
  - \\(X\_\\sigma\\)
    线性插值得到的加噪样本，是模型实际输入。
  - \\(v^\\star\\)
    训练监督目标，表示从 clean sample 指向 noise 的速度。
  - \\(\\hat v\_\\theta\\)
    模型预测的速度场；video/action 两个分支各预测自己的速度。

<!-- end list -->

    return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample, noise, timestep):
        return noise - sample

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F7. 训练中 video/action 的实际加噪** [fastwam.py:448-477](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ Z\_\\sigma=(1-\\sigma\_v)Z\_0+\\sigma\_v\\epsilon\_v,\\qquad A\_\\sigma=(1-\\sigma\_a)A\_0+\\sigma\_a\\epsilon\_a \\\] \\\[ v\_v^\\star=\\epsilon\_v-Z\_0,\\qquad v\_a^\\star=\\epsilon\_a-A\_0 \\\]

### 参数说明

  - \\(Z\_0,Z\_\\sigma\\)
    clean video latent 和加噪 video latent，对应 `input_latents` 与 `latents`。
  - \\(A\_0,A\_\\sigma\\)
    clean action 和加噪 action，对应 `action` 与 `noisy_action`。
  - \\(\\sigma\_v,\\sigma\_a\\)
    video 和 action 各自采样的噪声强度；两者可以不同。
  - \\(\\epsilon\_v,\\epsilon\_a\\)
    video latent 空间和 action 空间各自的高斯噪声。
  - \\(v\_v^\\star,v\_a^\\star\\)
    video flow target 和 action flow target，对应 `target_video` 与 `target_action`。

<!-- end list -->

    noise_video = torch.randn_like(input_latents)
    timestep_video = self.train_video_scheduler.sample_training_t(...)
    latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
    target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

    noise_action = torch.randn_like(action)
    timestep_action = self.train_action_scheduler.sample_training_t(...)
    noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F8. 首帧 causal 条件** [fastwam.py:340-468](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ Z\_\\sigma\[:, :, 0\] \\leftarrow Z\_0\[:, :, 0\] \\\]

当 `fuse_vae_embedding_in_latents=True` 时，训练会把第一帧 latent 保持为 clean first-frame latent。 后续 video loss 也会排除这个首帧 latent，让模型预测未来 latent 的 flow。

### 参数说明

  - \\(Z\_\\sigma\[:, :, 0\]\\)
    加噪 video latent 的第 0 个 latent frame。
  - \\(Z\_0\[:, :, 0\]\\)
    clean first-frame latent，来自输入视频第一帧。
  - \\(\\leftarrow\\)
    赋值操作，不是可学习运算；代码直接覆盖第 0 帧 latent。
  - 作用
    让模型在训练和推理时都以当前观测帧为条件预测未来，而不是重建当前帧。

<!-- end list -->

    first_frame_latents = input_latents[:, :, 0:1]
    ...
    if inputs["first_frame_latents"] is not None:
        latents[:, :, 0:1] = inputs["first_frame_latents"]

</div>

</div>

</div>

<div id="experts" class="section">

## 6\. Video Expert 与 Action Expert

两个 expert 都先把原始输入转成 transformer token。Video Expert 接收 VAE latent， 通过 3D patch embedding 得到时空 token；Action Expert 接收 noisy action， 通过线性层得到 action token。

<div class="formula-card">

<div class="formula-head">

**F9. Video patchify** [wan\_video\_dit.py:367-408, 555-600](../src/fastwam/models/wan22/wan_video_dit.py)

</div>

<div class="formula-body">

\\\[ \\tilde Z = \\operatorname{Conv3D}\_{p\_f,p\_h,p\_w}(Z\_\\sigma) \\in \\mathbb{R}^{B \\times D\_v \\times F' \\times h' \\times w'} \\\] \\\[ X\_v = \\operatorname{Flatten}\_{F'h'w'}(\\tilde Z) \\in \\mathbb{R}^{B \\times S\_v \\times D\_v},\\qquad S\_v=F'h'w' \\\]

当前配置中 \\(D\_v=3072\\)，patch size 为 \\(\[1,2,2\]\\)。

### 参数说明

  - \\(Z\_\\sigma\\)
    加噪 video latent，形状 \\(\[B,48,F,h,w\]\\)。
  - \\(p\_f,p\_h,p\_w\\)
    VideoDiT patch size；当前配置是 \\(\[1,2,2\]\\)，即时间维不降采样，空间维每 \\(2\\times2\\) 合成一个 token。
  - \\(\\tilde Z\\)
    3D 卷积 patch embedding 后的特征图。
  - \\(D\_v\\)
    Video Expert hidden dim，当前为 3072。
  - \\(F',h',w'\\)
    patchify 后的 latent 网格尺寸；\\(F'=F/p\_f\\)，\\(h'=h/p\_h\\)，\\(w'=w/p\_w\\)。
  - \\(S\_v\\)
    video token 总数，等于 \\(F'h'w'\\)。

<!-- end list -->

    self.patch_embedding = nn.Conv3d(
        in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

    x = self.patchify(x)
    x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F10. Action token 化** [action\_dit.py:74-85, 280-286](../src/fastwam/models/wan22/action_dit.py)

</div>

<div class="formula-body">

\\\[ X\_a = A\_\\sigma W\_a + b\_a \\in \\mathbb{R}^{B \\times T\_a \\times D\_a} \\\] \\\[ C\_a = \\operatorname{MLP}\_a(C) \\in \\mathbb{R}^{B \\times L\_c \\times D\_a} \\\]

当前配置中 \\(D\_a=1024\\)，文本/proprio context 在进入 Action Expert 后也被投影到 1024 维。

### 参数说明

  - \\(A\_\\sigma\\)
    加噪动作序列，形状 \\(\[B,T\_a,d\_a\]\\)。
  - \\(W\_a,b\_a\\)
    `action_encoder` 的线性层参数，把动作维度 \\(d\_a\\) 投到 \\(D\_a\\)。
  - \\(X\_a\\)
    Action Expert 的 residual stream token，形状 \\(\[B,T\_a,D\_a\]\\)。
  - \\(C\\)
    原始 context，维度是 4096；包含文本 token 和可选 proprio token。
  - \\(C\_a\\)
    Action Expert 内部使用的 context embedding，维度被投影到 \\(D\_a=1024\\)。
  - \\(L\_c\\)
    context token 数；加入 proprio 或 motion token 后会增加。

<!-- end list -->

    self.action_encoder = nn.Linear(action_dim, hidden_dim)
    self.text_embedding = nn.Sequential(
        nn.Linear(text_dim, hidden_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden_dim, hidden_dim),
    )

    tokens = self.action_encoder(action_tokens)
    context_emb = self.text_embedding(context)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F11. 时间调制参数** [action\_dit.py:280-286](../src/fastwam/models/wan22/action_dit.py)

</div>

<div class="formula-body">

\\\[ e\_t = \\operatorname{MLP}\_{time}(\\operatorname{SinCos}(t)),\\qquad m\_t = \\operatorname{MLP}\_{proj}(e\_t) \\\] \\\[ m\_t \\rightarrow (\\Delta\_{\\text{msa}}, S\_{\\text{msa}}, G\_{\\text{msa}}, \\Delta\_{\\text{mlp}}, S\_{\\text{mlp}}, G\_{\\text{mlp}}) \\\]

每层 DiT block 都使用这 6 组调制量控制 self-attention 和 MLP 的残差分支。

### 参数说明

  - \\(t\\)
    scheduler 采样得到的 timestep，video 和 action 分支各有自己的 timestep。
  - \\(\\operatorname{SinCos}(t)\\)
    一维正弦位置编码，把标量 timestep 转成 `freq_dim` 维向量。
  - \\(e\_t\\)
    time embedding MLP 输出，维度等于 expert hidden dim。
  - \\(m\_t\\)
    time projection 输出，会被拆成 6 份，每份都是 hidden dim。
  - \\(\\Delta,S,G\\)
    \\(\\Delta\\) 是 shift，\\(S\\) 是 scale，\\(G\\) 是 residual gate；分别调制 MSA 和 MLP 两个分支。

<!-- end list -->

    t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

</div>

</div>

</div>

<div id="attention" class="section">

## 7\. MoT 与 Cross Attention

FastWAM 的核心不是单个 transformer，而是每一层都同时处理 video expert 和 action expert。 MoT 先做 video/action token 的混合 self-attention，随后每个 expert 各自对 context 做 cross attention。

<div class="formula-card">

<div class="formula-head">

**F12. DiT block 的调制输入** [mot.py:58-75, 163-172](../src/fastwam/models/wan22/mot.py)

</div>

<div class="formula-body">

\\\[ U\_e^\\ell = \\operatorname{LN}\_1(X\_e^\\ell)\\odot(1+S\_{\\text{msa},e}) +\\Delta\_{\\text{msa},e} \\\] \\\[ Q\_e^\\ell=\\operatorname{RoPE}(\\operatorname{RMSNorm}(U\_e^\\ell W^Q\_e)), \\quad K\_e^\\ell=\\operatorname{RoPE}(\\operatorname{RMSNorm}(U\_e^\\ell W^K\_e)), \\quad V\_e^\\ell=U\_e^\\ell W^V\_e \\\]

这里 \\(e\\in\\{\\text{video},\\text{action}\\}\\)。两个 expert 使用各自的 projection，但 head 数和 head dim 必须一致。

### 参数说明

  - \\(e\\)
    expert 类型，取 video 或 action。
  - \\(\\ell\\)
    transformer block 层号；FastWAM 配置中两个 expert 都是 30 层。
  - \\(X\_e^\\ell\\)
    第 \\(\\ell\\) 层输入 token；video/action 各自有不同 hidden dim。
  - \\(U\_e^\\ell\\)
    做 Q/K/V 投影前的调制后 token。
  - \\(\\operatorname{LN}\_1\\)
    block 的第一个 LayerNorm，对应 `block.norm1`。
  - \\(\\odot\\)
    逐元素乘法，scale 和 gate 都是按 hidden dim 广播到所有 token。
  - \\(W\_e^Q,W\_e^K,W\_e^V\\)
    该 expert self-attention 的可学习 Q/K/V 线性投影。
  - \\(\\operatorname{RoPE}\\)
    旋转位置编码；video 使用 3D 时空 RoPE，action 使用 1D 时间 RoPE。

<!-- end list -->

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self._split_modulation(block, t_mod)
    attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

    q = block.self_attn.norm_q(block.self_attn.q(attn_input))
    k = block.self_attn.norm_k(block.self_attn.k(attn_input))
    v = block.self_attn.v(attn_input)
    q = rope_apply(q, freqs, block.num_heads)
    k = rope_apply(k, freqs, block.num_heads)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F13. MoT 混合 self-attention** [mot.py:518-553](../src/fastwam/models/wan22/mot.py)

</div>

<div class="formula-body">

\\\[ Q=\[Q\_v;Q\_a\],\\qquad K=\[K\_v;K\_a\],\\qquad V=\[V\_v;V\_a\] \\\] \\\[ O=\\operatorname{Attention}(Q,K,V;M\_{\\text{mot}}) = \\operatorname{Softmax}\\left(\\frac{QK^\\top}{\\sqrt{d\_h}} + M\_{\\text{mot}}\\right)V \\\] \\\[ O=\[O\_v;O\_a\] \\\]

代码把两个 expert 的 Q/K/V 沿 sequence 维拼接，做一次 masked attention，再按原 sequence 长度切回 video/action。

### 参数说明

  - \\(\[Q\_v;Q\_a\]\\)
    沿 token sequence 维拼接，不是特征维拼接。拼接后 sequence 长度是 \\(S\_v+S\_a\\)。
  - \\(Q,K,V\\)
    拼接后的 mixed attention 输入，内部形状会 reshape 为 \\(\[B,H,S,d\_h\]\\)。
  - \\(M\_{\\text{mot}}\\)
    video/action 之间的可见性 mask，决定哪些 query token 可以看哪些 key token。
  - \\(O\\)
    混合 attention 输出，再按原 token 长度切分成 \\(O\_v\\) 和 \\(O\_a\\)。
  - \\(d\_h\\)
    每个 attention head 的维度，当前为 128。

<!-- end list -->

    q_cat = torch.cat(q_chunks, dim=1)
    k_cat = torch.cat(k_chunks, dim=1)
    v_cat = torch.cat(v_chunks, dim=1)

    mixed = self._mixed_attention(
        q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

    mixed_slice = mixed[:, start:end, :]
    updated_tokens = self._apply_post_with_optional_checkpoint(...)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F14. MoT attention mask** [fastwam.py:385-407](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ M\_{\\text{mot}}= \\begin{bmatrix} M\_{v\\rightarrow v} & 0 \\\\ M\_{a\\rightarrow v\_0} & \\mathbf{1}\_{a\\rightarrow a} \\end{bmatrix} \\\]

当前 mask 让 action token 可以看 action token 和首帧 video token。 video token 不看 action token。首帧 causal video 模式下，首帧 video token 也不能看未来 video token。

### 参数说明

  - \\(M\_{v\\rightarrow v}\\)
    video query 到 video key 的 mask；由 `build_video_to_video_mask` 构造。
  - \\(0\\)
    右上角块，表示 video query 不能 attend 到 action key。
  - \\(M\_{a\\rightarrow v\_0}\\)
    action query 只能 attend 到首帧 video token，而不是未来 video token。
  - \\(\\mathbf{1}\_{a\\rightarrow a}\\)
    action query 可以 attend 到所有 action key。
  - 布尔语义
    在 PyTorch SDPA 里 `True` 表示允许 attention，`False` 表示屏蔽。

<!-- end list -->

    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
    mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(...)
    mask[video_seq_len:, video_seq_len:] = True
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    mask[video_seq_len:, :first_frame_tokens] = True

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F15. self-attention 残差更新** [mot.py:99-122](../src/fastwam/models/wan22/mot.py)

</div>

<div class="formula-body">

\\\[ \\bar X\_e^\\ell = X\_e^\\ell + G\_{\\text{msa},e}\\odot W^O\_e O\_e^\\ell \\\]

self-attention 分支带有 gate。这个 gate 来自 timestep modulation。

### 参数说明

  - \\(\\bar X\_e^\\ell\\)
    完成 mixed self-attention residual 后、cross attention 前的 token。
  - \\(G\_{\\text{msa},e}\\)
    self-attention 分支 gate，来自时间调制参数。
  - \\(W\_e^O\\)
    self-attention 输出投影，对应 `block.self_attn.o`。
  - \\(O\_e^\\ell\\)
    MoT mixed attention 输出中切回给当前 expert 的那一段。

<!-- end list -->

    x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F16. Cross attention 的 Q/K/V** [wan\_video\_dit.py:198-220](../src/fastwam/models/wan22/wan_video_dit.py)

</div>

<div class="formula-body">

\\\[ Q\_c=\\operatorname{RMSNorm}(\\operatorname{LN}\_3(\\bar X\_e^\\ell)W^Q\_c), \\quad K\_c=\\operatorname{RMSNorm}(C\_e W^K\_c), \\quad V\_c=C\_e W^V\_c \\\] \\\[ Y\_c=\\operatorname{Softmax}\\left(\\frac{Q\_cK\_c^\\top}{\\sqrt{d\_h}}+M\_c\\right)V\_c \\\] \\\[ \\tilde X\_e^\\ell = \\bar X\_e^\\ell + W^O\_cY\_c \\\]

对 Action Expert：\\(\\bar X\_a^\\ell\\in\\mathbb{R}^{B\\times T\_a\\times1024}\\)， \\(C\_a\\in\\mathbb{R}^{B\\times L\_c\\times1024}\\)。 q/k/v 先投影到 \\(24\\times128=3072\\)，再 reshape 成多头。

### 参数说明

  - \\(Q\_c\\)
    cross attention 的 query，来自当前 expert token；Action Expert 中来自动作 token。
  - \\(K\_c,V\_c\\)
    cross attention 的 key/value，来自 context token；不是来自图像像素。
  - \\(C\_e\\)
    投影到当前 expert hidden dim 后的 context。video 分支是 \\(C\_v\\)，action 分支是 \\(C\_a\\)。
  - \\(M\_c\\)
    context mask，形状通常是 \\(\[B,1,S\_e,L\_c\]\\)；控制每个 query 能看哪些 context token。
  - \\(Y\_c\\)
    从 context 读出的加权 value。
  - \\(\\tilde X\_e^\\ell\\)
    完成 cross attention residual 后、MLP 前的 token。
  - \\(W\_c^Q,W\_c^K,W\_c^V,W\_c^O\\)
    cross attention 自己的 Q/K/V/O 可学习线性层，与 self-attention 的线性层不是同一组参数。

<!-- end list -->

    self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
    self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
    self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
    self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)

    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(ctx))
    v = self.v(ctx)
    x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
    return self.o(x)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F17. 多头 attention reshape** [wan\_video\_dit.py:14-21](../src/fastwam/models/wan22/wan_video_dit.py)

</div>

<div class="formula-body">

\\\[ Q\_c\\in\\mathbb{R}^{B\\times S\\times (H d\_h)} \\rightarrow \\mathbb{R}^{B\\times H\\times S\\times d\_h} \\\]

这里 reshape 不改变数值，只把线性投影后的 3072 维拆成 24 个 128 维 head。

### 参数说明

  - \\(S\\)
    query token 数；action cross attention 中 \\(S=T\_a\\)，video cross attention 中 \\(S=S\_v\\)。
  - \\(H\\)
    attention head 数，当前为 24。
  - \\(d\_h\\)
    每个 head 的维度，当前为 128。
  - \\(Hd\_h\\)
    attention 内部总维度，当前为 3072。
  - `rearrange`
    只改变张量视图/排列；真正的升维来自前面的 `nn.Linear(hidden_dim, H*d_h)`。

<!-- end list -->

    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F18. MLP 残差更新** [mot.py:120-122](../src/fastwam/models/wan22/mot.py)

</div>

<div class="formula-body">

\\\[ X\_e^{\\ell+1} = \\tilde X\_e^\\ell + G\_{\\text{mlp},e}\\odot \\operatorname{FFN}\\left( \\operatorname{LN}\_2(\\tilde X\_e^\\ell)\\odot(1+S\_{\\text{mlp},e}) +\\Delta\_{\\text{mlp},e} \\right) \\\]

### 参数说明

  - \\(X\_e^{\\ell+1}\\)
    当前 DiT block 输出，作为下一层输入。
  - \\(\\operatorname{FFN}\\)
    两层 MLP，中间 GELU，对应 `block.ffn`。
  - \\(\\operatorname{LN}\_2\\)
    MLP 前的 LayerNorm，对应 `block.norm2`。
  - \\(S\_{\\text{mlp},e},\\Delta\_{\\text{mlp},e}\\)
    MLP 输入的 scale 和 shift。
  - \\(G\_{\\text{mlp},e}\\)
    MLP residual branch 的 gate。

<!-- end list -->

    mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = block.gate(x, gate_mlp, block.ffn(mlp_input))

</div>

</div>

</div>

<div id="loss" class="section">

## 8\. 训练 Loss

MoT 输出每个 expert 的更新后 tokens。Video Expert head 把 video tokens 还原成 VAE latent flow， Action Expert head 把 action tokens 还原成 action flow。两者都用 MSE 对齐 flow-matching target。

<div class="formula-card">

<div class="formula-head">

**F19. Head 输出** [fastwam.py:530-532](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ \\hat v\_v = H\_v(X\_v^L),\\qquad \\hat v\_a = H\_a(X\_a^L) \\\]

### 参数说明

  - \\(X\_v^L,X\_a^L\\)
    经过全部 \\(L\\) 层 MoT/DiT block 后的 video/action token。这里 \\(L\\) 是层数，不是 context 长度。
  - \\(H\_v\\)
    Video Expert 的输出 head，把 video token 还原为 VAE latent flow 形状。
  - \\(H\_a\\)
    Action Expert 的输出 head，把 action token 还原为 action\_dim 维 flow。
  - \\(\\hat v\_v,\\hat v\_a\\)
    模型预测的 video flow 和 action flow，对应 `pred_video` 与 `pred_action`。

<!-- end list -->

    pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
    pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F20. Video flow loss** [fastwam.py:409-447, 534-548](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ \\mathcal{L}\_v = \\mathbb{E}\_{b} \\left\[ w(t\_v^{(b)}) \\cdot \\frac{1}{|\\Omega\_b|} \\sum\_{i\\in\\Omega\_b} \\left\\| \\hat v\_{v,i}^{(b)} - v\_{v,i}^{\\star(b)} \\right\\|\_2^2 \\right\] \\\]

如果启用 clean first-frame latent，loss 会裁掉第一帧 latent，只监督未来 latent。 `image_is_pad` 会过滤 padding 帧。

### 参数说明

  - \\(\\mathcal{L}\_v\\)
    video 分支 flow-matching loss。
  - \\(b\\)
    batch 样本索引。
  - \\(w(t\_v^{(b)})\\)
    video scheduler 对当前 timestep 的 loss weight。
  - \\(\\Omega\_b\\)
    当前样本中有效 video latent 元素集合；padding 帧和可选首帧会被排除。
  - \\(i\\)
    video latent 中的元素索引，包含通道、时间、空间位置。
  - \\(v\_{v,i}^{\\star(b)}\\)
    video flow target 的第 \\(i\\) 个元素。

<!-- end list -->

    if inputs["first_frame_latents"] is not None:
        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]

    loss_video_per_sample = self._compute_video_loss_per_sample(...)
    video_weight = self.train_video_scheduler.training_weight(timestep_video)
    loss_video = (loss_video_per_sample * video_weight).mean()

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F21. Action flow loss** [fastwam.py:550-563](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ \\mathcal{L}\_a = \\mathbb{E}\_{b} \\left\[ w(t\_a^{(b)}) \\cdot \\frac{1}{|\\mathcal{T}\_b|} \\sum\_{\\tau\\in\\mathcal{T}\_b} \\left\\| \\hat v\_{a,\\tau}^{(b)} - v\_{a,\\tau}^{\\star(b)} \\right\\|\_2^2 \\right\] \\\]

`action_is_pad` 过滤 padding action step。每个动作维度先求均值，再对有效时间步平均。

### 参数说明

  - \\(\\mathcal{L}\_a\\)
    action 分支 flow-matching loss。
  - \\(w(t\_a^{(b)})\\)
    action scheduler 对当前 timestep 的 loss weight。
  - \\(\\mathcal{T}\_b\\)
    当前样本中有效 action 时间步集合，padding step 会被排除。
  - \\(\\tau\\)
    动作序列中的时间步索引。
  - \\(\\hat v\_{a,\\tau}^{(b)}\\)
    模型预测的第 \\(\\tau\\) 个动作 flow 向量。
  - \\(v\_{a,\\tau}^{\\star(b)}\\)
    目标动作 flow，即 \\(\\epsilon\_a-A\_0\\) 的第 \\(\\tau\\) 个向量。

<!-- end list -->

    action_loss_token = F.mse_loss(
        pred_action.float(), target_action.float(), reduction="none"
    ).mean(dim=2)

    valid = (~action_is_pad).to(...)
    action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
    action_weight = self.train_action_scheduler.training_weight(timestep_action)
    loss_action = (action_loss_per_sample * action_weight).mean()

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F22. 总 loss** [fastwam.py:563-568](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ \\mathcal{L} = \\lambda\_v\\mathcal{L}\_v+\\lambda\_a\\mathcal{L}\_a \\\]

### 参数说明

  - \\(\\mathcal{L}\\)
    最终反向传播的总 loss。
  - \\(\\lambda\_v,\\lambda\_a\\)
    video/action loss 权重，对应配置里的 `loss.lambda_video` 和 `loss.lambda_action`。
  - \\(\\mathcal{L}\_v,\\mathcal{L}\_a\\)
    分别来自 video latent flow 和 action flow 的 MSE loss。

<!-- end list -->

    loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action

</div>

</div>

</div>

<div id="inference" class="section">

## 9\. 推理动作生成

`infer_action` 只需要当前图像、prompt 或预计算 context、可选 proprio。 它先把当前图像编码成首帧 latent，再缓存 video 分支每层的 K/V。之后每个 action denoise step 只更新 action 分支。

<div class="formula-card">

<div class="formula-head">

**F23. 动作初始化** [fastwam.py:952-962](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ A\_{\\sigma=1}^{(0)} \\sim \\mathcal{N}(0,I), \\qquad Z\_{\\text{first}}=\\operatorname{VAEEnc}(I\_0) \\\]

### 参数说明

  - \\(A\_{\\sigma=1}^{(0)}\\)
    推理初始 action latent，完全由高斯噪声采样；上标 \\((0)\\) 表示第 0 个推理迭代状态。
  - \\(\\mathcal{N}(0,I)\\)
    标准正态分布，形状是 \\(\[1,\\text{action\_horizon},d\_a\]\\)。
  - \\(I\_0\\)
    当前观测图像，即推理时传入的第一帧图像。
  - \\(Z\_{\\text{first}}\\)
    当前图像经过 VAE 编码得到的首帧 latent。

<!-- end list -->

    latents_action = torch.randn((1, action_horizon, self.action_expert.action_dim), ...)
    first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F24. Video cache** [fastwam.py:993-1022](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ \\mathcal{K}\\mathcal{V}\_v = \\operatorname{PrefillVideoCache}(Z\_{\\text{first}}, C) \\\]

缓存的是每层 video self-attention 的 K/V。推理动作时 action queries 会 attend 到 cached video K/V 和当前 action K/V。

### 参数说明

  - \\(\\mathcal{K}\\mathcal{V}\_v\\)
    video 分支每一层 self-attention 的 key/value cache。
  - \\(\\operatorname{PrefillVideoCache}\\)
    先只跑 video 分支，把首帧条件对应的 per-layer video K/V 保存下来。
  - \\(Z\_{\\text{first}}\\)
    首帧 VAE latent，推理 action 时固定不变。
  - \\(C\\)
    推理条件 context，来自 prompt 编码或外部传入的 precomputed context，再加可选 proprio。

<!-- end list -->

    video_pre = self.video_expert.pre_dit(
        x=first_frame_latents, timestep=timestep_video,
        context=context, context_mask=context_mask, action=None,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    video_kv_cache = self.mot.prefill_video_cache(...)

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F25. 推理积分步** [scheduler\_continuous.py:63-88](../src/fastwam/models/wan22/schedulers/scheduler_continuous.py)

</div>

<div class="formula-body">

\\\[ \\sigma\_0\>\\sigma\_1\>\\cdots\>\\sigma\_K=0,\\qquad \\Delta\_k=\\sigma\_{k+1}-\\sigma\_k \\\] \\\[ A\_{\\sigma\_{k+1}} = A\_{\\sigma\_k} + \\Delta\_k \\hat v\_\\theta(A\_{\\sigma\_k}, Z\_{\\text{first}}, C, t\_k) \\\]

因为 \\(\\Delta\_k\<0\\)，模型预测的是 noise-minus-clean 方向，积分时会从噪声逐步回到 clean action。

### 参数说明

  - \\(\\sigma\_k\\)
    第 \\(k\\) 个推理噪声强度，从接近 1 递减到 0。
  - \\(\\Delta\_k\\)
    相邻两个噪声强度的差，等于 \\(\\sigma\_{k+1}-\\sigma\_k\\)，因此通常是负数。
  - \\(A\_{\\sigma\_k}\\)
    第 \\(k\\) 步当前 action sample。
  - \\(\\hat v\_\\theta\\)
    FastWAM action 分支在当前 step 预测的 action flow。
  - \\(K\\)
    推理总步数，对应 `num_inference_steps`，不要和 attention 的 key \\(K\\) 混淆。

<!-- end list -->

    u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    sigma_steps = self._phi(u_steps, shift)
    timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
    deltas = sigma_steps[1:] - sigma_steps[:-1]

    return sample + model_output * delta

</div>

</div>

<div class="formula-card">

<div class="formula-head">

**F26. Action with cached video K/V** [mot.py:343-445](../src/fastwam/models/wan22/mot.py)

</div>

<div class="formula-body">

\\\[ K=\[K\_v^{\\text{cache}};K\_a\],\\qquad V=\[V\_v^{\\text{cache}};V\_a\] \\\] \\\[ O\_a=\\operatorname{Attention}(Q\_a,K,V;M\_{a}) \\\]

这一步是动作推理的关键加速：当前图像对应的 video K/V 不随 action denoise step 变化，所以只算一次。

### 参数说明

  - \\(Q\_a\\)
    当前 action token 生成的 query，每个 denoise step 都要重新计算。
  - \\(K\_v^{\\text{cache}},V\_v^{\\text{cache}}\\)
    从首帧 video 分支缓存下来的 key/value。
  - \\(K\_a,V\_a\\)
    当前 action token 生成的 key/value。
  - \\(M\_a\\)
    action query 行对应的 joint attention mask，只允许 action 看首帧 video token 和 action token。
  - \\(O\_a\\)
    action mixed self-attention 输出，之后还会进入 action cross attention 和 MLP。

<!-- end list -->

    k_cat = torch.cat([k_video, k_action], dim=1)
    v_cat = torch.cat([v_video, v_action], dim=1)
    mixed = self._mixed_attention(
        q_cat=q_action, k_cat=k_cat, v_cat=v_cat,
        attention_mask=action_attention_mask,
    )

</div>

</div>

</div>

<div id="zipmotion" class="section">

## 10\. 接入 ZipMotion 的位置

如果把 ZipMotion 作为额外 motion context 接入 FastWAM，最小侵入路径是扩展 context，而不是改 VAE 或 MoT 主干。 原因是 FastWAM 已经有稳定的 cross attention 通道负责让 action/video token 读取外部条件。

<div class="formula-card">

<div class="formula-head">

**F27. Motion tokens 作为 context 扩展** [fastwam.py:219-240](../src/fastwam/models/wan22/fastwam.py)

</div>

<div class="formula-body">

\\\[ M\_z = E\_{\\text{ZipMotion}}(I\_0,\\operatorname{tracks}) \\in \\mathbb{R}^{B\\times L\_m\\times d\_z} \\\] \\\[ C\_m = M\_z W\_m + b\_m \\in \\mathbb{R}^{B\\times L\_m\\times4096} \\\] \\\[ C'=\\operatorname{Concat}(C\_{\\text{text}},c\_p,C\_m) \\\]

这样 Action Expert 和 Video Expert 原有的 cross attention 公式不变，只是 context 长度从 \\(L\_c\\) 变成 \\(L\_c+L\_m\\)。 训练时不能使用未来真实轨迹作为 policy 输入，否则会产生信息泄漏；应使用当前可观测帧预测出的 motion tokens。

### 参数说明

  - \\(E\_{\\text{ZipMotion}}\\)
    ZipMotion/ZipMo 的点追踪 motion encoder 或 predictor。
  - \\(I\_0\\)
    当前可观测图像帧；部署时只能使用当前和历史观测，不能使用未来帧。
  - \\(\\operatorname{tracks}\\)
    由当前可用观测预测或构造的点轨迹信息；如果来自未来 GT 轨迹，会造成 policy 信息泄漏。
  - \\(M\_z\\)
    ZipMotion 输出的 motion token 序列，原始维度是 \\(d\_z\\)。
  - \\(W\_m,b\_m\\)
    motion projector，把 \\(d\_z\\) 投到 FastWAM 原始 context 维度 4096。
  - \\(C\_m\\)
    可被 FastWAM cross attention 读取的 motion context token。
  - \\(C'\\)
    扩展后的总 context；后续会被各 expert 的 `text_embedding` 投到自己的 hidden dim。

<!-- end list -->

    # 伪代码：与 proprio append 类似
    motion_context = self.motion_projector(zipmotion_tokens)  # [B, Lm, 4096]
    motion_mask = torch.ones((B, Lm), dtype=torch.bool, device=context.device)
    context = torch.cat([context, motion_context], dim=1)
    context_mask = torch.cat([context_mask, motion_mask], dim=1)

</div>

</div>

</div>

<div id="source-index" class="section">

## 11\. 源码索引

| 模块                       | 路径                                                                                                                             | 本文使用的关键位置                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| FastWAM 主流程              | [src/fastwam/models/wan22/fastwam.py](../src/fastwam/models/wan22/fastwam.py)                                                  | context/proprio、build\_inputs、training\_loss、infer\_action |
| Action Expert            | [src/fastwam/models/wan22/action\_dit.py](../src/fastwam/models/wan22/action_dit.py)                                           | action encoder、context projection、ActionDiT pre\_dit       |
| Video Expert 与 attention | [src/fastwam/models/wan22/wan\_video\_dit.py](../src/fastwam/models/wan22/wan_video_dit.py)                                    | Conv3D patchify、CrossAttention、flash\_attention、DiTBlock   |
| MoT                      | [src/fastwam/models/wan22/mot.py](../src/fastwam/models/wan22/mot.py)                                                          | video/action QKV 拼接、mixed self-attention、video cache       |
| Flow scheduler           | [src/fastwam/models/wan22/schedulers/scheduler\_continuous.py](../src/fastwam/models/wan22/schedulers/scheduler_continuous.py) | 采样时间、加噪、训练目标、推理积分                                          |
| LeRobot 视频数据集            | [src/fastwam/datasets/lerobot/robot\_video\_dataset.py](../src/fastwam/datasets/lerobot/robot_video_dataset.py)                | 多相机拼接、文本 context cache、action/proprio 对齐                   |

</div>

</div>

</div>
