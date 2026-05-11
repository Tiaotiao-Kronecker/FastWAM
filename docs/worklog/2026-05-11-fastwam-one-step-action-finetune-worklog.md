# 工作记录：FastWAM one-step action fine-tune

日期：2026-05-11

## 动机

按照 `docs/plan/2026-05-10-fastwam-one-step-diffusion-plan.md` 的 Phase 1，先实现 action-only one-step endpoint fine-tune。第一性原理上，FastWAM 的 action 推理如果要从多步 flow-matching 压缩到单步，需要训练模型在高噪声端一次预测 action endpoint；因此本轮先固定 video expert，只更新 action/MoT 路径，避免同时改变 video 生成分布。

## 实现内容

新增 action-only one-step 训练模型：

- `src/fastwam/models/wan22/fastwam_one_step_action.py`
- `src/fastwam/runtime.py` 中新增 `create_fastwam_one_step_action`

新增配置：

- `configs/model/fastwam_one_step_action.yaml`
- `configs/task/robotwin_one_step_action_3cam_384_1e-4.yaml`

训练器调整：

- `src/fastwam/trainer.py` 支持无 Deepspeed plugin 的单卡 smoke run。
- optimizer 只收集 `requires_grad=True` 的参数。
- 初始化 optimizer 前调用 `configure_trainable_parameters()`，当前 one-step 版本冻结 video expert。

本地忽略规则：

- `.gitignore` 增加 `.conda/`，避免把本地 conda 环境纳入 Git。

## 训练目标

本轮使用 action flow-matching 的固定高噪声 timestep：

```text
x_t = (1 - sigma) * action + sigma * noise
target_velocity = noise - action
pred_endpoint = x_t - sigma * pred_velocity
loss = 0.5 * MSE(pred_velocity, target_velocity) + 0.5 * MSE(pred_endpoint, action)
```

action padding 通过 `action_is_pad` mask 排除，不让 padding token 影响 loss。

## 环境与资产

本地环境：

- Python 环境：`.conda/fastwam`
- PyTorch：`2.7.1+cu128`
- 本地安装：`pip install -e .`

基础资产：

- release checkpoint：`checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`
- RoboTwin 数据：`data/robotwin2.0/`
- VAE：`checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`
- text encoder：`checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors`
- tokenizer：`checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/`

由于训练配置关闭在线 text encoder 加载，已预先生成固定 instruction 的文本 embedding cache：

- cache：`data/text_embeds_cache/robotwin`
- instruction：`Grab the smooth green plastic bottle and lift it with the left arm`

## 验证命令

配置解析：

```bash
.conda/fastwam/bin/python scripts/train.py task=robotwin_one_step_action_3cam_384_1e-4 --cfg job
```

1 step smoke run：

```bash
CUDA_VISIBLE_DEVICES=1 \
DIFFSYNTH_DOWNLOAD_SOURCE=modelscope \
DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk2/wangchen/projects/FastWAM/checkpoints \
TOKENIZERS_PARALLELISM=false \
.conda/fastwam/bin/python scripts/train.py \
  task=robotwin_one_step_action_3cam_384_1e-4 \
  max_steps=1 save_every=1 log_every=1 eval_every=0 \
  output_dir=./runs/robotwin_one_step_action_smoke \
  '+data.train.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm' \
  '+data.val.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm'
```

10 step fine-tune check：

```bash
CUDA_VISIBLE_DEVICES=1 \
DIFFSYNTH_DOWNLOAD_SOURCE=modelscope \
DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk2/wangchen/projects/FastWAM/checkpoints \
TOKENIZERS_PARALLELISM=false \
.conda/fastwam/bin/python scripts/train.py \
  task=robotwin_one_step_action_3cam_384_1e-4 \
  max_steps=10 save_every=10 log_every=1 eval_every=0 \
  output_dir=./runs/robotwin_one_step_action_10step \
  '+data.train.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm' \
  '+data.val.override_instruction=Grab the smooth green plastic bottle and lift it with the left arm'
```

## 当前结果

- 配置解析通过。
- 1 step smoke run 通过，loss 约 `0.0030`。
- smoke 权重保存到 `runs/robotwin_one_step_action_smoke/checkpoints/weights/step_000001.pt`。
- 10 step fine-tune check 通过，没有 NaN 或 OOM。
- 10 step 最终 loss：`0.1320`，其中 `loss_action_endpoint=0.0661`，`loss_action_velocity=0.0659`。
- 10 step 速度约 `0.60 step/s`。
- 10 step 权重保存到 `runs/robotwin_one_step_action_10step/checkpoints/weights/step_000010.pt`。
- 10 step 训练 state 保存到 `runs/robotwin_one_step_action_10step/checkpoints/state/step_000010`。

保存 state 时 Accelerate 输出了 shared tensor removal warning。该 warning 来自 `dit` 和 `mot` 共享参数命名，日志提示通常可接受；后续若要从 full state resume，应优先做一次 reload 验证。当前 weight-only checkpoint 已成功落盘，可用于下一步评测接入。

## 后续

1. 进入更长步数的 action-only fine-tune。
2. 将 fine-tuned checkpoint 接入 `num_inference_steps=1` 评测，与原 release checkpoint 的 1-step sampler baseline 对比。
3. 若 action-only 结果优于原 1-step sampler，再考虑 teacher-student 蒸馏。
