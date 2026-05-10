# 工作记录：One-Step Diffusion Phase 0 启动

日期：2026-05-10

## 动机

按照 `docs/plan/2026-05-10-fastwam-one-step-diffusion-plan.md` 开始执行 Phase 0。第一性原理上，one-step 方案是否值得继续，必须先得到现有 checkpoint 在不同推理步数下的退化曲线；否则直接进入训练或蒸馏会缺少可比较的下界。

## 前置检查

已检查：

- `git status -sb`
- `checkpoints/`
- `data/`
- `evaluate_results/`
- `scripts/`
- `experiments/robotwin/run_robotwin_manager.py`
- `experiments/libero/run_libero_manager.py`

结果：

- 当前分支：`main`，相对 `tiaotiao/main` ahead 1。
- 当前本地没有 `checkpoints/` 目录。
- 当前本地没有 `data/` 目录。
- 当前本地没有 `evaluate_results/` 目录。
- 因缺少 checkpoint、dataset stats 和数据/评测环境，无法直接运行 RoboTwin 或 LIBERO baseline。

## 本次执行

新增 Phase 0 baseline 脚本：

- `scripts/run_one_step_diffusion_baseline.sh`

用途：

- 对同一 checkpoint 依次运行 `num_inference_steps=1 2 4 10`。
- 支持 `robotwin` 和 `libero` 两个 benchmark。
- 通过环境变量固定 `CKPT`、`STATS`、`NUM_GPUS`、`MAX_TASKS_PER_GPU`、`STEPS`、`RUN_ID`、`OUTPUT_ROOT`。
- 对 checkpoint 和 stats 文件做显式存在性检查，避免长任务启动后才失败。
- 每个步数的输出目录名包含 `RUN_ID`，避免重复执行时覆盖结果。
- RoboTwin 默认创建 `experiments/robotwin/fastwam_policy` 到 `third_party/RoboTwin/policy/fastwam_policy` 的软链接。

示例：

```bash
CKPT=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
STATS=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
NUM_GPUS=8 \
STEPS="1 2 4 10" \
bash scripts/run_one_step_diffusion_baseline.sh robotwin
```

## 验证

- 已执行 `bash -n scripts/run_one_step_diffusion_baseline.sh`，语法检查通过。
- 已执行 `chmod +x scripts/run_one_step_diffusion_baseline.sh`。
- 未运行真实评测，因为本地缺少 checkpoint、dataset stats、数据和 benchmark 环境。

## 下一步

1. 补齐 release checkpoint、dataset stats、数据和 benchmark 环境。
2. 运行 RoboTwin baseline：`STEPS="1 2 4 10"`。
3. 将每组 `summary.csv`、`summary.json` 和失败任务记录归档到本 worklog 或新的实验 worklog。
4. 如果 1-step baseline 退化可接受，再进入 action-only one-step endpoint 微调；如果退化严重，优先做 teacher-student 蒸馏设计。
