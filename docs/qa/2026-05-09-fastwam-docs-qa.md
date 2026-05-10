# 问答归档：FastWAM 项目内文档与 One-Step

日期：2026-05-09

## Q1：是否已经有 `worklog` 这个 skill？

没有。当前本地 skills 里只有 `imagegen`、`openai-docs`、`plugin-creator`、`skill-creator`、`skill-installer`，没有 `worklog/SKILL.md`。

## Q2：还需要创建全局 `worklog` skill 吗？

不需要。当前处理方式改为在 FastWAM 项目内创建项目级 `agent.md` 和 `docs/` 归档目录。

## Q3：FastWAM 后续文档归档到哪里？

统一归档到仓库根目录下的 `docs/`：

- `docs/worklog/`：工作记录。
- `docs/qa/`：问答归档。
- `docs/plan/`：计划归档。

具体规则见 `agent.md`。

## Q4：这里的 one-step 如何理解？

本次按“把 FastWAM 常用准备、训练、评测入口整理成可直接复用的命令模板”理解。当前不新增脚本，先归纳已有入口：

- `scripts/preprocess_action_dit_backbone.py`
- `scripts/precompute_text_embeds.py`
- `scripts/train_zero1.sh`
- `experiments/libero/run_libero_manager.py`
- `experiments/robotwin/run_robotwin_manager.py`

详细归纳见 `docs/plan/2026-05-09-fastwam-one-step-plan.md`。
