#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.conda/fastwam/bin/python"}
RUN_TAG=${RUN_TAG:-shortcut_official_1_pilot2k_20260513}
TRAIN_GPU=${TRAIN_GPU:-4}
EVAL_GPUS=${EVAL_GPUS:-4}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}
NUM_TRIALS=${NUM_TRIALS:-50}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}
POLL_SECONDS=${POLL_SECONDS:-60}

TRAIN_OUTPUT=${TRAIN_OUTPUT:-"./runs/libero_one_step_shortcut_official_pilot2k_20260513"}
CKPT=${CKPT:-"$TRAIN_OUTPUT/checkpoints/weights/step_002000.pt"}
EVAL_OUTPUT=${EVAL_OUTPUT:-"./evaluate_results/libero/$RUN_TAG"}
TASK_CONFIG=${TASK_CONFIG:-libero_one_step_shortcut_official_2cam224_1e-4}
EVAL_MANAGER_SESSION=${EVAL_MANAGER_SESSION:-"${RUN_TAG}_eval_manager"}
EVAL_WORKER_SESSION=${EVAL_WORKER_SESSION:-"${RUN_TAG}_eval_workers"}

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

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

count_results() {
    if [ ! -d "$EVAL_OUTPUT" ]; then
        echo 0
        return
    fi
    find "$EVAL_OUTPUT" -type f -name "gpu*_task*_results.json" | wc -l
}

if [ ! -f "$CKPT" ]; then
    log "Starting official-style shortcut fine-tune: output=$TRAIN_OUTPUT gpu=$TRAIN_GPU"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON_BIN" scripts/train.py \
        task=libero_one_step_shortcut_official_2cam224_1e-4 \
        output_dir="$TRAIN_OUTPUT" \
        max_steps=2000 \
        save_every=1000 \
        log_every=10 \
        wandb.enabled=false
else
    log "Training checkpoint already exists: $CKPT"
fi

if [ ! -f "$CKPT" ]; then
    log "Missing expected checkpoint after training: $CKPT"
    exit 2
fi

if [ -f "$EVAL_OUTPUT/summary.csv" ] && [ "$(count_results)" -ge 40 ]; then
    log "Eval already complete: $EVAL_OUTPUT"
    cat "$EVAL_OUTPUT/summary.csv"
    exit 0
fi

if ! tmux has-session -t "$EVAL_MANAGER_SESSION" 2>/dev/null; then
    log "Starting eval: output=$EVAL_OUTPUT ckpt=$CKPT gpus=$EVAL_GPUS workers_per_gpu=$WORKERS_PER_GPU"
    TASK_CONFIG="$TASK_CONFIG" \
    CKPT="$CKPT" \
    OUTPUT_DIR="$EVAL_OUTPUT" \
    RUN_TAG="$RUN_TAG" \
    NUM_INFERENCE_STEPS=1 \
    NUM_TRIALS="$NUM_TRIALS" \
    SAVE_ROLLOUT_VIDEO="$SAVE_ROLLOUT_VIDEO" \
    GPUS="$EVAL_GPUS" \
    WORKERS_PER_GPU="$WORKERS_PER_GPU" \
    MANAGER_SESSION_NAME="$EVAL_MANAGER_SESSION" \
    WORKER_SESSION_NAME="$EVAL_WORKER_SESSION" \
    experiments/libero/run_libero_multiworker_batch.sh
else
    log "Eval manager already running: $EVAL_MANAGER_SESSION"
fi

while tmux has-session -t "$EVAL_MANAGER_SESSION" 2>/dev/null; do
    log "Eval running: results=$(count_results)/40"
    sleep "$POLL_SECONDS"
done

if [ -s "$EVAL_OUTPUT/failed_tasks.txt" ]; then
    log "Eval has failures: $EVAL_OUTPUT/failed_tasks.txt"
    cat "$EVAL_OUTPUT/failed_tasks.txt"
    exit 3
fi

if [ -f "$EVAL_OUTPUT/summary.csv" ]; then
    log "Eval complete: $EVAL_OUTPUT"
    cat "$EVAL_OUTPUT/summary.csv"
else
    log "Eval manager ended before summary.csv was written: $EVAL_OUTPUT"
    exit 4
fi
