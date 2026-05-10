# 问答归档：FastWAM One-Step Diffusion

日期：2026-05-10

## Q1：当前是否已经有 one-step diffusion 方案归档？

此前没有单独归档。已有 `docs/plan/2026-05-09-fastwam-one-step-plan.md` 记录的是一步式命令工作流，不是 one-step diffusion 技术方案。

## Q2：把 `num_inference_steps=1` 算不算 one-step diffusion？

只能算 1-step sampler baseline，不算真正训练好的 one-step diffusion 模型。

当前 scheduler 已支持 `num_inference_steps=1`，可以直接跑，但模型训练时是随机 timestep 的 flow-matching 目标，默认评测是 10 steps。没有经过单步蒸馏或 endpoint 专门训练时，直接 1 step 大概率会有质量和成功率退化。

## Q3：FastWAM 这里应该叫 diffusion 还是 flow-matching？

代码实现更准确地说是 continuous flow-matching。

关键点：

- 加噪：`x_t = (1 - sigma) * x_0 + sigma * noise`
- 训练目标：`noise - x_0`
- 推理更新：`x <- x + v * delta`

“one-step diffusion” 可以作为讨论简称，但实现和公式应按 flow-matching 来做。

## Q4：应该先压缩 video 还是 action？

建议先做 action-only。

原因：

- FastWAM 评测最终关心机器人 action success。
- `FastWAM.infer_action` 路径最轻，只需要首帧 video cache。
- video 单步生成质量不稳定时会干扰 action。
- action-only baseline 更容易定位收益和失败原因。

## Q5：直接监督训练和 teacher-student 蒸馏哪个先做？

先做直接监督的 action-only endpoint 微调，再做 teacher-student。

推荐顺序：

1. 直接用数据集 action ground truth 训练单步 endpoint。
2. 跑通后与原模型 `num_inference_steps=1/2/4/10` 做对照。
3. 如果直接监督不足，再用 10-step teacher 蒸馏 student。

这样可以先验证代码路径、loss、mask、checkpoint 兼容性，避免一开始就引入昂贵 teacher。

## Q6：one-step action loss 应该怎么定义？

给定 action ground truth `a_0` 和噪声 `eps`：

- 固定高噪声端 `a_1 = eps`。
- 模型输入 noisy action 和 timestep。
- 模型预测 velocity `v_pred`。
- flow target 是 `v_target = eps - a_0`。
- endpoint 预测可写成 `a_pred = eps - v_pred`。

loss 可以组合：

- `MSE(v_pred, v_target)`
- `MSE(a_pred, a_0)`
- 使用 `action_is_pad` mask 忽略 padding token。

## Q7：是否需要修改 scheduler？

第一阶段不需要大改 scheduler。

当前 `WanContinuousFlowMatchScheduler.build_inference_schedule(num_inference_steps=1)` 已能产生单步 schedule。更关键的是增加训练分支，让模型真正适应单步 endpoint。后续如果需要更精细的 sigma / shift 搜索，再扩展 config。

## Q8：哪些代码位置最相关？

主要位置：

- `src/fastwam/models/wan22/schedulers/scheduler_continuous.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/models/wan22/fastwam_idm.py`
- `src/fastwam/runtime.py`
- `configs/model/*.yaml`
- `configs/task/*.yaml`
- `configs/train.yaml`
- `experiments/libero/run_libero_manager.py`
- `experiments/robotwin/run_robotwin_manager.py`

## Q9：如何判断 one-step 方案是否值得继续？

至少比较四组：

- 原 checkpoint，`num_inference_steps=10`
- 原 checkpoint，`num_inference_steps=4`
- 原 checkpoint，`num_inference_steps=1`
- one-step fine-tuned checkpoint，`num_inference_steps=1`

如果 one-step fine-tuned 在速度明显提升的同时，success rate 明显优于原 checkpoint 的 1-step sampler，并接近 4-step 或 10-step，则值得继续做 teacher-student 蒸馏。

## Q10：最大风险是什么？

主要风险：

- 单步从纯噪声到 action endpoint 太难，直接监督可能不够。
- action 分布多模态，MSE 可能过度平均。
- video/action joint 分支一起压缩时，video 质量会影响 action。
- teacher-student cache 如果 seed/noise/context 没对齐，会得到不可复现或错误监督。
- action normalization 和 padding mask 处理错误会导致训练看似下降但评测失败。
