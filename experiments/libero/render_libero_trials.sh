#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.conda/fastwam/bin/python"}
GPU_ID=${GPU_ID:-0}
TASK_CONFIG=${TASK_CONFIG:-libero_uncond_2cam224_1e-4}
TASK_SUITE=${TASK_SUITE:-libero_spatial}
TASK_ID=${TASK_ID:-0}
TRIAL_INDICES=${TRIAL_INDICES:-"[0]"}
NUM_TRIALS=${NUM_TRIALS:-50}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-./evaluate_results/libero/video_reruns}

CKPT=${CKPT:-./checkpoints/fastwam_release/libero_uncond_2cam224.pt}
DATASET_STATS=${DATASET_STATS:-./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/DATA/disk3/tmp/libero_config}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export HF_HOME=${HF_HOME:-/DATA/disk3/tmp/hf_home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/DATA/disk3/tmp/xdg_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/DATA/disk3/tmp/matplotlib-fastwam}
export DIFFSYNTH_DOWNLOAD_SOURCE=${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"$ROOT_DIR/checkpoints"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" experiments/libero/eval_libero_single.py \
  task="$TASK_CONFIG" \
  ckpt="$CKPT" \
  gpu_id=0 \
  EVALUATION.task_suite_name="$TASK_SUITE" \
  EVALUATION.task_id="$TASK_ID" \
  EVALUATION.num_trials="$NUM_TRIALS" \
  EVALUATION.num_inference_steps="$NUM_INFERENCE_STEPS" \
  EVALUATION.dataset_stats_path="$DATASET_STATS" \
  EVALUATION.output_dir="$OUTPUT_DIR" \
  EVALUATION.save_rollout_video=true \
  EVALUATION.trial_indices="$TRIAL_INDICES"

echo "Rendered LIBERO rollout videos into: $OUTPUT_DIR/$TASK_SUITE/videos"
