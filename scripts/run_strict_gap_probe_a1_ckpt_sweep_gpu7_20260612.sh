#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.conda/fastwam/bin/python"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}

cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/DATA/disk3/tmp/libero_config}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export HF_HOME=${HF_HOME:-/DATA/disk3/cache/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/DATA/disk3/tmp/xdg_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/DATA/disk3/tmp/matplotlib-fastwam}
export DIFFSYNTH_DOWNLOAD_SOURCE=${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"$ROOT_DIR/checkpoints"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export TMUX_GRID_ROWS=${TMUX_GRID_ROWS:-1}
export TMUX_GRID_COLS=${TMUX_GRID_COLS:-2}
export MONITORING_INTERVAL=${MONITORING_INTERVAL:-10}
export STATUS_INTERVAL=${STATUS_INTERVAL:-60}

TASK_FILE=experiments/libero/task_sets/libero_gap_probe_v1.txt
TASK_CONFIG=libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5
NUM_TRIALS=50
NUM_INFERENCE_STEPS=1
WORKERS_PER_GPU=1
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false

A1_RUN_DIR="$ROOT_DIR/runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000"
A1_WEIGHTS_DIR="$A1_RUN_DIR/checkpoints/weights"
A1_DATASET_STATS="$A1_RUN_DIR/dataset_stats.json"

QUEUE_LOG="$ROOT_DIR/evaluate_results/libero/a1_ckpt_sweep_strict_gap_probe_gpu7_${RUN_STAMP}.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "Missing required file: $path"
    exit 1
  fi
}

run_eval() {
  local step="$1"
  local tag="a1_step${step}"
  local ckpt="$A1_WEIGHTS_DIR/step_${step}.pt"

  require_file "$ckpt"
  require_file "$A1_DATASET_STATS"
  require_file "$TASK_FILE"

  local output_dir="./evaluate_results/libero/${tag}_strict_gap_probe_v1_steps${NUM_INFERENCE_STEPS}_50trials_gpu7_${RUN_STAMP}"
  local worker_session="${tag}_strict_gap_probe_workers_gpu7_${RUN_STAMP}"

  if [[ -e "$output_dir" ]]; then
    log "Refusing to reuse existing output directory: $output_dir"
    exit 1
  fi

  export SESSION_NAME="$worker_session"

  log "START $tag"
  log "  task_config=$TASK_CONFIG"
  log "  ckpt=$ckpt"
  log "  dataset_stats=$A1_DATASET_STATS"
  log "  output_dir=$output_dir"
  log "  num_inference_steps=$NUM_INFERENCE_STEPS"
  log "  worker_session=$worker_session"

  "$PYTHON_BIN" experiments/libero/run_libero_manager.py \
    task="$TASK_CONFIG" \
    ckpt="$ckpt" \
    seed=42 \
    EVALUATION.dataset_stats_path="$A1_DATASET_STATS" \
    EVALUATION.num_trials="$NUM_TRIALS" \
    EVALUATION.trial_indices=null \
    EVALUATION.num_inference_steps="$NUM_INFERENCE_STEPS" \
    EVALUATION.save_rollout_video="$SAVE_ROLLOUT_VIDEO" \
    EVALUATION.save_action_trace="$SAVE_ACTION_TRACE" \
    EVALUATION.output_dir="$output_dir" \
    +EVALUATION.policy_subprocess=false \
    MULTIRUN.task_file="$TASK_FILE" \
    MULTIRUN.num_gpus=1 \
    MULTIRUN.max_tasks_per_gpu="$WORKERS_PER_GPU" \
    2>&1 | tee -a "$QUEUE_LOG"

  log "DONE $tag"
}

log "A1 checkpoint strict gap-probe sweep starting"
log "RUN_STAMP=$RUN_STAMP"
log "ROOT_DIR=$ROOT_DIR"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "QUEUE_LOG=$QUEUE_LOG"
log "TASK_FILE=$TASK_FILE"
log "NUM_TRIALS=$NUM_TRIALS"
log "SAVE_ACTION_TRACE=$SAVE_ACTION_TRACE"
log "SAVE_ROLLOUT_VIDEO=$SAVE_ROLLOUT_VIDEO"

for step in 050000 055000 060000 065000 070000; do
  run_eval "$step"
done

log "A1 checkpoint strict gap-probe sweep finished"
