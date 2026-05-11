# 工作记录：FastWAM 运行资产下载

日期：2026-05-11

## 动机

Phase 0 one-step diffusion baseline 需要本地 checkpoint、dataset stats 和 benchmark 数据。第一性原理上，评测输入必须先满足三个条件：模型权重可加载、数据路径符合配置、评测输出目录可由脚本自动生成。

## 已完成

### Release checkpoints

目标目录：

- `checkpoints/fastwam_release/`

已下载并校验大小：

- `libero_uncond_2cam224.pt`：`12041735140` bytes
- `libero_uncond_2cam224_dataset_stats.json`：`40939` bytes
- `robotwin_uncond_3cam_384.pt`：`12041813092` bytes
- `robotwin_uncond_3cam_384_dataset_stats.json`：`88715` bytes

说明：

- 初始使用 `hf download` 时，大 checkpoint 下载多次停滞。
- 登录 Hugging Face 后仍遇到 `429`/Xet token 限流和断点续传不稳定。
- 最终通过 Git LFS 克隆 `https://huggingface.co/yuanty/fastwam`，分别拉取两个 `.pt` 文件，再复制到项目 checkpoint 目录。
- 下载完成后已删除临时 LFS 仓库和失败的 HF cache。

### LIBERO data

目标目录：

- `data/libero_mujoco3.3.2/`

已下载并解压：

- `libero_10_no_noops_lerobot.tar.gz`
- `libero_goal_no_noops_lerobot.tar.gz`
- `libero_object_no_noops_lerobot.tar.gz`
- `libero_spatial_no_noops_lerobot.tar.gz`

解压后确认存在：

- `data/libero_mujoco3.3.2/libero_10_no_noops_lerobot`
- `data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot`
- `data/libero_mujoco3.3.2/libero_object_no_noops_lerobot`
- `data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot`

### RoboTwin data

目标目录：

- `data/robotwin2.0/`

已下载并解压：

- `robotwin2.0.tar.gz.part-00` 到 `robotwin2.0.tar.gz.part-07`
- `dataset_stats.json`

解压后确认存在：

- `data/robotwin2.0/robotwin2.0/data`
- `data/robotwin2.0/robotwin2.0/meta`
- `data/robotwin2.0/robotwin2.0/videos`

说明：

- RoboTwin 压缩分卷约 79GB。
- 使用 Git LFS 克隆 `https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam`，拉取分卷后复制到项目 `data/robotwin2.0/`。
- 已删除临时 Git LFS 仓库 `/tmp/fastwam_robotwin_lfs`。

## 当前占用

执行后检查：

- `checkpoints/fastwam_release`：约 `23G`
- `data/libero_mujoco3.3.2`：约 `8.8G`
- `data/robotwin2.0`：约 `149G`

`data/robotwin2.0` 当前包含压缩分卷和解压后的数据。压缩分卷可在确认不需要复用后清理。

## 验证

- `git status --short` 输出为空，大文件目录未进入 Git 跟踪。
- RoboTwin 解压结构符合 README 预期：`data/`、`meta/`、`videos/`。
- LIBERO 解压结构符合 README 预期：四个 `*_lerobot` 目录。

## 下一步

1. 按 `scripts/run_one_step_diffusion_baseline.sh` 运行 Phase 0 baseline。
2. RoboTwin 示例：

```bash
CKPT=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
STATS=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
NUM_GPUS=8 \
STEPS="1 2 4 10" \
bash scripts/run_one_step_diffusion_baseline.sh robotwin
```

3. 若评测环境依赖缺失，先补 RoboTwin / LIBERO 仿真环境，不再重复下载模型和数据。
