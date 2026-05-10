# 工作记录：FastWAM 文档归档与 One-Step 归纳

日期：2026-05-09

## 背景

用户确认不需要创建全局 `worklog` skill，希望在当前 FastWAM 项目内建立项目级 agent 文档约定，并创建 `docs/` 目录，将工作记录、问答、计划分别归档，方便后续调用。

## 已检查

- 项目目录：`/DATA/disk2/wangchen/projects/FastWAM`
- 原先无 `docs/` 目录。
- 原先无 `AGENTS.md` 或 `agent.md`。
- 参考了以下文件归纳 FastWAM one-step 工作流：
  - `README_zh.md`
  - `README.md`
  - `configs/sim_libero.yaml`
  - `configs/sim_robotwin.yaml`
  - `scripts/train_zero1.sh`
  - `experiments/libero/run_libero_manager.py`
  - `experiments/robotwin/run_robotwin_manager.py`

## 本次新增

- `agent.md`
  - 规定 `docs/worklog`、`docs/qa`、`docs/plan` 的用途。
  - 规定文档命名和更新规则。
  - 记录 FastWAM one-step 归纳文档入口。
- `docs/README.md`
  - 作为 `docs/` 目录索引。
- `docs/plan/2026-05-09-fastwam-one-step-plan.md`
  - 归纳模型准备、T5 embedding cache、训练、LIBERO/RoboTwin release 评测、自训练 checkpoint 评测的命令模板。
- `docs/worklog/2026-05-09-fastwam-docs-worklog.md`
  - 记录本次文档体系创建工作。
- `docs/qa/2026-05-09-fastwam-docs-qa.md`
  - 记录本次关于是否创建 skill、如何归档、one-step 如何理解的问答。

## One-Step 摘要

FastWAM 当前 one-step 入口主要是对现有脚本和 manager 的命令模板化：

- ActionDiT backbone：`scripts/preprocess_action_dit_backbone.py`
- 文本 embedding cache：`scripts/precompute_text_embeds.py`
- 训练：`scripts/train_zero1.sh`
- LIBERO 评测：`experiments/libero/run_libero_manager.py`
- RoboTwin 评测：`experiments/robotwin/run_robotwin_manager.py`

详细命令见 `docs/plan/2026-05-09-fastwam-one-step-plan.md`。

## 验证

- 已执行 `git status --short`，当前新增文件为 `agent.md` 和 `docs/`。
- 已执行 `find docs -maxdepth 3 -type f | sort`，确认新增文档都在预期目录下。
- 已抽查 `agent.md` 和 `docs/plan/2026-05-09-fastwam-one-step-plan.md` 内容。
- 本次只整理文档，不运行训练或评测。
