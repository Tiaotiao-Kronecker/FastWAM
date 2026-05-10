# FastWAM Agent Notes

本文件记录 FastWAM 项目内给后续 agent 使用的协作约定。优先遵循仓库现有代码风格、README 和配置文件；本文件只补充项目本地的文档归档规则。

## Docs Layout

所有后续可复用的上下文统一归档到 `docs/`：

- `docs/worklog/`：工作记录。记录已做事项、涉及文件、运行命令、验证结果、遗留问题。
- `docs/qa/`：问答归档。记录用户问题、结论、关键依据，方便后续快速检索。
- `docs/plan/`：计划归档。记录目标、执行步骤、当前状态、后续动作。

建议文件命名：

```text
docs/<category>/YYYY-MM-DD-short-topic.md
```

其中 `<category>` 取 `worklog`、`qa` 或 `plan`。文档默认使用中文；命令、路径、配置项、错误信息保持原文。

## Update Rules

- 只要完成了非平凡修改或排查，就更新或新增一条 `docs/worklog/`。
- 用户提出的问题、关键决策或容易反复问到的结论，归档到 `docs/qa/`。
- 涉及多步任务、实验流程、训练/评测路线时，归档到 `docs/plan/`。
- 文档中引用代码时使用仓库相对路径，例如 `configs/sim_robotwin.yaml`。
- 不把大文件、数据集、checkpoint、日志全文塞进 `docs/`；只记录路径、命令和结论。
- 如果任务继续推进，应优先查阅最近的 `docs/worklog/` 和对应 `docs/plan/`。

## FastWAM One-Step Context

当前已归纳的 FastWAM one-step 工作流见：

- `docs/plan/2026-05-09-fastwam-one-step-plan.md`
- `docs/worklog/2026-05-09-fastwam-docs-worklog.md`
- `docs/qa/2026-05-09-fastwam-docs-qa.md`

这里的 one-step 指把常用训练、预处理和评测入口收敛到可直接复制执行的命令，主要基于 `README_zh.md`、`configs/sim_libero.yaml`、`configs/sim_robotwin.yaml` 和 `scripts/train_zero1.sh`。
