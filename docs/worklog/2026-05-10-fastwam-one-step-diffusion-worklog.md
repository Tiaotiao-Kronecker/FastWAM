# 工作记录：FastWAM One-Step Diffusion 归档

日期：2026-05-10

## 背景

用户要求把此前关于添加 one-step diffusion 的问答、讨论、计划进行合理分析并归档。此前项目内只有一步式命令工作流归档，没有 one-step diffusion 技术方案归档。

## 已检查

阅读了项目归档约定：

- `agent.md`

阅读和检索了当前 diffusion / flow-matching 相关实现：

- `src/fastwam/models/wan22/schedulers/scheduler_continuous.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/models/wan22/fastwam_idm.py`
- `src/fastwam/runtime.py`
- `configs/train.yaml`
- `configs/sim_libero.yaml`
- `configs/sim_robotwin.yaml`
- `configs/model/fastwam.yaml`
- `configs/model/fastwam_joint.yaml`
- `configs/model/fastwam_idm.yaml`

关键观察：

- 当前是 continuous flow-matching，而不是离散 DDPM/DDIM。
- scheduler 已支持 `num_inference_steps=1`。
- 默认训练评测步数是 `eval_num_inference_steps=10`。
- 当前训练 loss 是随机 timestep 的 velocity target，不是 one-step endpoint 专门训练。
- action-only 路径比 video/action joint 路径更适合作为第一阶段。

## 本次新增/更新

新增：

- `docs/plan/2026-05-10-fastwam-one-step-diffusion-plan.md`
- `docs/qa/2026-05-10-fastwam-one-step-diffusion-qa.md`
- `docs/worklog/2026-05-10-fastwam-one-step-diffusion-worklog.md`

更新：

- `agent.md`
- `docs/README.md`

## 归档结论

one-step diffusion 后续应拆成两层：

1. 先跑现有 checkpoint 的 `num_inference_steps=1` baseline，确认不用训练时的性能下界。
2. 再做 action-only one-step endpoint 微调或蒸馏，避免直接同时压缩 video + action。

第一阶段不建议重写 scheduler，也不建议引入外部 diffusion 框架。应沿用当前 flow-matching 公式和推理接口，只增加明确的 one-step action 训练分支与配置。

## 验证状态

本次只做文档归档，没有运行训练或评测。

已完成：

- 抽查 `docs/plan/2026-05-10-fastwam-one-step-diffusion-plan.md`。
- 抽查 `docs/qa/2026-05-10-fastwam-one-step-diffusion-qa.md`。
- 抽查本 worklog。
- 执行 `git status --short`，确认改动只包含本次文档新增和索引更新。
