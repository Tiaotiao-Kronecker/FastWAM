#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

LOG_FILE=${LOG_FILE:-./evaluate_results/libero/a1_step010000_three_group_gap_probe_20260604/serial_after_current_residual_clip.log}
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

SERIAL_GPUS=${SERIAL_GPUS:-4}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}
NUM_TRIALS=${NUM_TRIALS:-50}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-1}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}
SAVE_ACTION_TRACE=${SAVE_ACTION_TRACE:-false}
TASK_SET=${TASK_SET:-libero_gap_probe_v1}
DATASET_STATS=${DATASET_STATS:-./runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/dataset_stats.json}
EXTRA_MANAGER_ARGS=${EXTRA_MANAGER_ARGS:-+EVALUATION.policy_subprocess=true}

CURRENT_RESIDUAL_CLIP_MANAGER=${CURRENT_RESIDUAL_CLIP_MANAGER:-libero_residual_clip025_step010000_gap_manager_20260604}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

wait_for_tmux_session_to_finish() {
  local session=$1
  local label=$2

  if tmux has-session -t "$session" 2>/dev/null; then
    log "Waiting for existing $label session to finish: $session"
    while tmux has-session -t "$session" 2>/dev/null; do
      sleep 60
    done
    log "Existing $label session finished: $session"
  else
    log "Existing $label session is not present, continuing: $session"
  fi
}

run_group_and_wait() {
  local label=$1
  local task_config=$2
  local ckpt=$3

  export TASK_CONFIG="$task_config"
  export CKPT="$ckpt"
  export GPUS="$SERIAL_GPUS"
  export RUN_TAG="a1_step010000_${label}_gap_probe_serial_v1_steps1_50trials_20260604"
  export OUTPUT_DIR="./evaluate_results/libero/${RUN_TAG}"
  export MANAGER_SESSION_NAME="libero_${label}_step010000_gap_serial_manager_20260604"
  export WORKER_SESSION_NAME="libero_${label}_step010000_gap_serial_workers_20260604"

  log "Starting serial eval for $label on GPU(s): $SERIAL_GPUS"
  bash experiments/libero/run_libero_task_set_eval.sh

  log "Waiting for manager to finish: $MANAGER_SESSION_NAME"
  while tmux has-session -t "$MANAGER_SESSION_NAME" 2>/dev/null; do
    sleep 60
  done

  if [[ -s "$OUTPUT_DIR/failed_tasks.txt" ]]; then
    log "Serial eval for $label finished with failures:"
    sed 's/^/[failed] /' "$OUTPUT_DIR/failed_tasks.txt"
  else
    log "Serial eval for $label finished with no manager-recorded failures."
  fi
}

wait_for_tmux_session_to_finish "$CURRENT_RESIDUAL_CLIP_MANAGER" "residual_clip025"

run_group_and_wait \
  control \
  libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5 \
  ./runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_continue_control_weights_from070k_10k_20260603/checkpoints/weights/step_010000.pt

run_group_and_wait \
  residual_only \
  libero_one_step_meanflow_a1_residual_only_lora_eqanchor_2cam224_5e-5 \
  ./runs/libero_one_step_meanflow_a1_residual_only_lora_eqanchor_2cam224_5e-5/a1_2_residual_only_weights_from070k_10k_20260604/checkpoints/weights/step_010000.pt

log "Serial eval queue finished."
