# FastWAM One-Step Diffusion 计划归档

日期：2026-05-10

## 归档说明

本文件整理此前关于 “one-step diffusion” 的问答、讨论和计划，并结合当前 FastWAM 代码做分析性归纳。这里不逐字复述原始对话，而是沉淀后续可执行的技术路线。

项目内更准确的实现语境是 continuous flow-matching：`WanContinuousFlowMatchScheduler` 使用 `x_t = (1 - sigma) * x_0 + sigma * noise`，训练目标是 `noise - x_0`，推理时用 Euler 更新 `x <- x + v * delta`。文档中沿用 “one-step diffusion” 作为讨论简称。

## 当前代码基线

相关文件：

- `src/fastwam/models/wan22/schedulers/scheduler_continuous.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/models/wan22/fastwam_idm.py`
- `configs/train.yaml`
- `configs/sim_libero.yaml`
- `configs/sim_robotwin.yaml`
- `configs/model/fastwam.yaml`
- `configs/model/fastwam_joint.yaml`
- `configs/model/fastwam_idm.yaml`

现状：

- 默认训练评测里 `eval_num_inference_steps=10`。
- LIBERO/RoboTwin 评测可通过 `EVALUATION.num_inference_steps` 覆盖推理步数。
- scheduler 已支持 `num_inference_steps=1`，因此可以直接跑 1-step sampler 基线。
- 当前训练 loss 是随机连续 timestep 的 flow-matching loss，不是专门为单步从纯噪声到数据端点训练的学生模型。
- `FastWAM.infer_action` 是最轻的一条路径：只编码首帧 video token cache，再迭代 denoise action。
- `FastWAMJoint` / `FastWAMIDM` 的 action 会依赖更多 video latent，上线 one-step 时需要先决定 video 是否也被单步压缩。

## 关键结论

1. `num_inference_steps=1` 可以作为第一基线，但不等于已经训练好的 one-step diffusion 模型。
2. 真正的 one-step 模型需要训练或蒸馏，让模型适配从高噪声端一次预测到 action/video endpoint 的分布。
3. 优先做 action-only one-step，比同时压缩 video + action 风险更低，也更贴合 FastWAM 的评测目标。
4. 如果只追求机器人动作成功率，未来视频质量可以先作为辅助信号，而不是第一阶段优化目标。
5. 代码中是 flow-matching 目标，后续实现应避免把 DDPM/DDIM 的公式直接套进来。

## 推荐路线

### Phase 0：无训练基线

目标：量化现有 checkpoint 直接减少采样步数的退化曲线。

推荐使用脚本：

```bash
CKPT={ckpt_path} \
STATS={stats_path} \
NUM_GPUS={num_gpus} \
STEPS="1 2 4 10" \
bash scripts/run_one_step_diffusion_baseline.sh robotwin
```

也可以手动运行单组对照：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt={ckpt_path} \
  EVALUATION.dataset_stats_path={stats_path} \
  EVALUATION.num_inference_steps=1 \
  MULTIRUN.num_gpus={num_gpus}
```

建议同时跑：

```text
num_inference_steps = 1, 2, 4, 10
```

记录指标：

- success rate
- per-task success rate
- latency / action chunk 时间
- 是否出现 action 爆炸、夹爪异常、任务早期失败
- seed、checkpoint、dataset stats、评测配置

预期：1-step sampler 可能明显掉点，但它给后续蒸馏提供必须的 baseline。

执行状态：

- 2026-05-10：已新增 `scripts/run_one_step_diffusion_baseline.sh`。
- 2026-05-10：当前本地缺少 `checkpoints/` 和 `data/`，尚未实际跑评测。

### Phase 1：action-only endpoint 微调

目标：先把 action denoising 训练成单步 endpoint predictor。

推荐从 `FastWAM.infer_action` 对应路径开始：

- video 只使用首帧 latent cache。
- action 从纯噪声 `a_1` 出发。
- 固定 `t_action` 到高噪声端，例如 `num_train_timesteps` 对应的 sigma 近似 1。
- 模型输出 `v = noise - action_gt`。
- endpoint 可由 `action_pred = noise - v_pred` 得到。
- loss 同时包含 velocity MSE 和 endpoint/action MSE，并继续使用 `action_is_pad` mask。

建议第一版先冻结或低学习率更新 video expert，主训练 action expert + MoT 中 action 相关参数，降低显存和稳定性风险。

候选文件改动：

- 新增 model config：`configs/model/fastwam_one_step_action.yaml`
- 新增 task config：`configs/task/*one_step_action*.yaml`
- 在模型中增加 one-step action loss 分支，或新增 `FastWAMOneStepAction` 子类。
- 在 `runtime.py` 增加对应 factory。
- 在 trainer 日志里记录 `loss_action_endpoint`、`loss_action_velocity`。

### Phase 2：teacher-student 蒸馏

目标：让 one-step student 逼近原多步 teacher 的输出分布，而不是只拟合数据集 ground truth。

Teacher：

- 使用当前 release 或已训练 checkpoint。
- 使用 10-step 或质量较稳的步数生成 action chunk。
- 固定同一初始 noise / seed，保存 teacher action endpoint。

Student：

- 从同一初始 noise 单步预测 action endpoint。
- loss 组合：
  - `MSE(student_action, teacher_action)`
  - 可选 `MSE(student_action, action_gt)`
  - 可选 velocity loss，稳定训练。

缓存策略：

- 小规模先 online teacher 验证正确性。
- 大规模训练时做 offline cache，避免每个 batch 都跑 10-step teacher。
- cache 至少记录 prompt/context、first frame 标识、seed/noise 生成方式、teacher action、checkpoint tag。

### Phase 3：joint / IDM one-step 扩展

目标：如果 action-only 不够，再把 video branch 也纳入单步压缩。

两条路线：

- `FastWAMJoint`：video/action 同步从噪声一步到 endpoint，action 可 attend 到生成的 video token。
- `FastWAMIDM`：先 one-step video，再用 denoised video 作为 cond，one-step action。

风险：

- video latent 单步质量不足会污染 action。
- 同时压缩 video 和 action 会增加训练不稳定性。
- 如果最终评测主要看 action success，video 端不应过早成为主瓶颈。

## 最小实现建议

第一版不要直接大改 scheduler。保持 scheduler 的 one-step schedule 能跑，并把训练目标改成显式 endpoint/action one-step loss。

优先实现：

1. 增加 action-only one-step 训练模式。
2. 新增配置，不影响现有 `fastwam`、`fastwam_joint`、`fastwam_idm`。
3. 用小 batch 跑通 forward/backward。
4. 在同一 checkpoint 上比较 `num_inference_steps=1` 与 one-step fine-tuned 模型。
5. 只在确认 action-only 有收益后再做 teacher-student 或 joint video/action。

## 验证清单

- `num_inference_steps=1` 原模型评测可正常跑通。
- one-step action loss 的 mask 与 `action_is_pad` 对齐。
- endpoint 反推 `action_pred = noise - v_pred` 的 shape、dtype、device 正确。
- 训练保存和加载 checkpoint 不影响旧模型。
- LIBERO/RoboTwin manager 可通过 config 选择 one-step checkpoint。
- 同一 seed 下 one-step student 输出稳定可复现。

## 暂不做的事

- 暂不把 “one-step diffusion” 做成全局 skill。
- 暂不修改现有 release 评测默认步数。
- 暂不把 video 和 action 一起蒸馏作为第一阶段目标。
- 暂不引入外部 diffusion 框架，优先沿用当前 flow-matching scheduler 与模型结构。

## 后续入口

后续如果开始实现，建议先从一个小 PR/commit 做：

- `FastWAM` action-only one-step loss 分支。
- 新 model/task config。
- 一个最小 smoke test 或 dry-run 脚本。
- 一份新的 `docs/worklog/YYYY-MM-DD-fastwam-one-step-diffusion-implementation.md` 记录实验结果。
