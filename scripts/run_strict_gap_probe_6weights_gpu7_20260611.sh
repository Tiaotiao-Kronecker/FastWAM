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
NUM_TRIALS=50
WORKERS_PER_GPU=1
SAVE_ACTION_TRACE=true
SAVE_ROLLOUT_VIDEO=false

QUEUE_LOG="$ROOT_DIR/evaluate_results/libero/strict_gap_probe_6weights_gpu7_${RUN_STAMP}.log"
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
  local tag="$1"
  local task_config="$2"
  local ckpt="$3"
  local dataset_stats="$4"
  local num_inference_steps="$5"

  require_file "$ckpt"
  require_file "$dataset_stats"
  require_file "$TASK_FILE"

  local output_dir="./evaluate_results/libero/${tag}_strict_gap_probe_v1_steps${num_inference_steps}_50trials_gpu7_${RUN_STAMP}"
  local worker_session="${tag}_strict_gap_probe_workers_gpu7_${RUN_STAMP}"

  if [[ -e "$output_dir" ]]; then
    log "Refusing to reuse existing output directory: $output_dir"
    exit 1
  fi

  export SESSION_NAME="$worker_session"

  log "START $tag"
  log "  task_config=$task_config"
  log "  ckpt=$ckpt"
  log "  dataset_stats=$dataset_stats"
  log "  output_dir=$output_dir"
  log "  num_inference_steps=$num_inference_steps"
  log "  worker_session=$worker_session"

  "$PYTHON_BIN" experiments/libero/run_libero_manager.py \
    task="$task_config" \
    ckpt="$ckpt" \
    seed=42 \
    EVALUATION.dataset_stats_path="$dataset_stats" \
    EVALUATION.num_trials="$NUM_TRIALS" \
    EVALUATION.trial_indices=null \
    EVALUATION.num_inference_steps="$num_inference_steps" \
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

log "Strict gap-probe queue starting"
log "RUN_STAMP=$RUN_STAMP"
log "ROOT_DIR=$ROOT_DIR"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "QUEUE_LOG=$QUEUE_LOG"

run_eval \
  "release1" \
  "libero_uncond_2cam224_1e-4" \
  "$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt" \
  "$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json" \
  1

run_eval \
  "release10" \
  "libero_uncond_2cam224_1e-4" \
  "$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt" \
  "$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json" \
  10

run_eval \
  "a1_step070000" \
  "libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/checkpoints/weights/step_070000.pt" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/dataset_stats.json" \
  1

run_eval \
  "a1_full_control_step080000" \
  "libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_full_continue_control_from070k_to080k_20260606_101145/checkpoints/weights/step_080000.pt" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_full_continue_control_from070k_to080k_20260606_101145/dataset_stats.json" \
  1

run_eval \
  "a1_2_full_residual_only_step070300" \
  "libero_one_step_meanflow_a1_residual_only_lora_eqanchor_2cam224_5e-5" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_residual_only_lora_eqanchor_2cam224_5e-5/a1_2_full_residual_only_from070k_to70300_smoke_20260606_190440/checkpoints/weights/step_070300.pt" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_residual_only_lora_eqanchor_2cam224_5e-5/a1_2_full_residual_only_from070k_to70300_smoke_20260606_190440/dataset_stats.json" \
  1

run_eval \
  "a1_3_full_clip025_step070300" \
  "libero_one_step_meanflow_a1_residual_clip_lora_eqanchor_2cam224_5e-5" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_residual_clip_lora_eqanchor_2cam224_5e-5/a1_3_full_clip025_from070k_to70300_smoke_20260606_192046/checkpoints/weights/step_070300.pt" \
  "$ROOT_DIR/runs/libero_one_step_meanflow_a1_residual_clip_lora_eqanchor_2cam224_5e-5/a1_3_full_clip025_from070k_to70300_smoke_20260606_192046/dataset_stats.json" \
  1

log "Strict gap-probe queue finished"
