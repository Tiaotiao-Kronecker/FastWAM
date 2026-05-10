# FastWAM One-Step 工作流计划

日期：2026-05-09

## 目标

把 FastWAM 当前常用的准备、训练和评测入口归纳成后续可以直接调用的 one-step 工作流。这里的 one-step 不是新增脚本，而是先把已有入口整理成固定命令模板，方便后续决定是否再封装成脚本。

## 前置条件

1. 环境已安装：

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

2. 模型目录已设置：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

3. 数据集和 checkpoint 已按 `README_zh.md` 放到对应目录。

## One-Step 模板

### 1. 预生成 ActionDiT backbone

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### 2. 预计算 T5 embedding cache

LIBERO：

```bash
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

RoboTwin：

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

多卡模板：

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### 3. 训练

LIBERO：

```bash
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
```

RoboTwin：

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

说明：

- `scripts/train_zero1.sh` 会自动生成 `RUN_ID`，输出到 `runs/<task>/<run_id>/`。
- 多机训练时脚本读取 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。
- 首次训练新任务时，先将对应 `configs/data/*.yaml` 的 `pretrained_norm_stats` 设为 `null`；生成 `dataset_stats.json` 后再回填。

### 4. 评测 release LIBERO checkpoint

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

关键配置：

- 默认评测 suite 来自 `configs/sim_libero.yaml` 的 `MULTIRUN.task_suite_names`。
- `EVALUATION.num_trials` 默认是 `50`。
- 输出目录默认在 `evaluate_results/libero/...`。

### 5. 评测 release RoboTwin checkpoint

先创建 policy 软链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

再运行评测：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

关键配置：

- 默认 `EVALUATION.instruction_type=unseen`。
- 默认 `EVALUATION.skip_get_obs_within_replan=true`，速度更快，但保存视频低帧率。
- manager 会同时跑 clean 和 randomized phase，并输出 `summary.csv`、`summary.json`。

### 6. 使用自己训练的 checkpoint 评测

LIBERO：

```bash
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

RoboTwin：

```bash
python experiments/robotwin/run_robotwin_manager.py task={task_name} ckpt={ckpt_path}
```

常用 `task_name`：

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

## 当前状态

- 已确认仓库原先没有 `docs/`、`AGENTS.md` 或 `agent.md`。
- 已将文档归档目录固定为 `docs/worklog`、`docs/qa`、`docs/plan`。
- 已基于现有 README、配置和入口脚本归纳 one-step 命令模板。

## 后续可选动作

- 如果这些模板稳定，可以继续新增 `scripts/one_step_*.sh` 包装脚本。
- 如果本地数据和 checkpoint 路径不同，后续应在 `docs/worklog/` 记录实际路径。
- 如果评测或训练失败，记录最小复现命令、日志路径和结论到 `docs/worklog/`。
