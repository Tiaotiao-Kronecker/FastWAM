#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.conda/fastwam/bin/python"}
GPUS=${GPUS:-0,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
NUM_TRIALS=${NUM_TRIALS:-50}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-./evaluate_results/libero/release_baseline_20260512_steps10_multiworker}
MANAGER_SESSION_NAME=${MANAGER_SESSION_NAME:-${SESSION_NAME:-libero_release10_multiworker}}
WORKER_SESSION_NAME=${WORKER_SESSION_NAME:-libero_workers_multiworker}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}

CKPT=${CKPT:-./checkpoints/fastwam_release/libero_uncond_2cam224.pt}
DATASET_STATS=${DATASET_STATS:-./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}
TASK_CONFIG=${TASK_CONFIG:-libero_uncond_2cam224_1e-4}

export CUDA_VISIBLE_DEVICES="$GPUS"
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
export PYTHON_BIN
export SESSION_NAME="$WORKER_SESSION_NAME"

MANAGER_CMD="export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; \
export HYDRA_FULL_ERROR=$HYDRA_FULL_ERROR; \
export LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH; \
export MUJOCO_GL=$MUJOCO_GL; \
export PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM; \
export HF_HOME=$HF_HOME; \
export XDG_CACHE_HOME=$XDG_CACHE_HOME; \
export MPLCONFIGDIR=$MPLCONFIGDIR; \
export DIFFSYNTH_DOWNLOAD_SOURCE=$DIFFSYNTH_DOWNLOAD_SOURCE; \
export DIFFSYNTH_MODEL_BASE_PATH=$DIFFSYNTH_MODEL_BASE_PATH; \
export TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM; \
export PYTHON_BIN=$PYTHON_BIN; \
export SESSION_NAME=$WORKER_SESSION_NAME; \
exec $PYTHON_BIN experiments/libero/run_libero_manager.py \
    task=$TASK_CONFIG \
    ckpt=$CKPT \
    EVALUATION.dataset_stats_path=$DATASET_STATS \
    EVALUATION.num_trials=$NUM_TRIALS \
    EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
    EVALUATION.save_rollout_video=$SAVE_ROLLOUT_VIDEO \
    EVALUATION.output_dir=$OUTPUT_DIR \
    MULTIRUN.create_only=false \
    MULTIRUN.num_gpus=$NUM_GPUS \
    MULTIRUN.max_tasks_per_gpu=$WORKERS_PER_GPU"

tmux new-session -d -s "$MANAGER_SESSION_NAME" -c "$ROOT_DIR" \
  "$MANAGER_CMD"

echo "Started manager tmux session: $MANAGER_SESSION_NAME"
echo "Worker tmux session name: $WORKER_SESSION_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS, workers per GPU: $WORKERS_PER_GPU"
