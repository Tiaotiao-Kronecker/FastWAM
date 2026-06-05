#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.conda/fastwam/bin/python"}
GPUS=${GPUS:-0,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
NUM_TRIALS=${NUM_TRIALS:-50}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-1}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}
SAVE_ACTION_TRACE=${SAVE_ACTION_TRACE:-false}
TRIAL_INDICES=${TRIAL_INDICES:-null}
EXTRA_MANAGER_ARGS=${EXTRA_MANAGER_ARGS:-}

TASK_SET=${TASK_SET:-libero_long_horizon_v1}
TASK_FILE=${TASK_FILE:-"experiments/libero/task_sets/${TASK_SET}.txt"}
TASK_SET_NAME=$(basename "$TASK_FILE" .txt)

TASK_CONFIG=${TASK_CONFIG:-libero_uncond_2cam224_1e-4}
CKPT=${CKPT:-./checkpoints/fastwam_release/libero_uncond_2cam224.pt}
DATASET_STATS=${DATASET_STATS:-./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}
RUN_TAG=${RUN_TAG:-"${TASK_SET_NAME}_steps${NUM_INFERENCE_STEPS}"}
OUTPUT_DIR=${OUTPUT_DIR:-"./evaluate_results/libero/${RUN_TAG}"}

MANAGER_SESSION_NAME=${MANAGER_SESSION_NAME:-${SESSION_NAME:-"${RUN_TAG}_manager"}}
WORKER_SESSION_NAME=${WORKER_SESSION_NAME:-"${RUN_TAG}_workers"}

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Task file not found: $TASK_FILE" >&2
  exit 1
fi

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
    EVALUATION.trial_indices=$TRIAL_INDICES \
    EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS \
    EVALUATION.save_rollout_video=$SAVE_ROLLOUT_VIDEO \
    EVALUATION.save_action_trace=$SAVE_ACTION_TRACE \
    EVALUATION.output_dir=$OUTPUT_DIR \
    MULTIRUN.task_file=$TASK_FILE \
    MULTIRUN.num_gpus=$NUM_GPUS \
    MULTIRUN.max_tasks_per_gpu=$WORKERS_PER_GPU \
    $EXTRA_MANAGER_ARGS"

tmux new-session -d -s "$MANAGER_SESSION_NAME" -c "$ROOT_DIR" \
  "$MANAGER_CMD"

echo "Started manager tmux session: $MANAGER_SESSION_NAME"
echo "Worker tmux session name: $WORKER_SESSION_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Task config: $TASK_CONFIG"
echo "Task file: $TASK_FILE"
echo "Checkpoint: $CKPT"
echo "GPUs: $GPUS, workers per GPU: $WORKERS_PER_GPU, trials: $NUM_TRIALS, steps: $NUM_INFERENCE_STEPS"
echo "Trial indices: $TRIAL_INDICES"
echo "Save rollout video: $SAVE_ROLLOUT_VIDEO"
echo "Save action trace: $SAVE_ACTION_TRACE"
echo "Extra manager args: $EXTRA_MANAGER_ARGS"
