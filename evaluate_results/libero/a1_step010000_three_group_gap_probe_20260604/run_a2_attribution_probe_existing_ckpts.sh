#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

export WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}
export NUM_TRIALS=${NUM_TRIALS:-50}
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-1}
export SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}
export SAVE_ACTION_TRACE=${SAVE_ACTION_TRACE:-false}
export TASK_SET=${TASK_SET:-libero_a2_attribution_probe_v1}
export DATASET_STATS=${DATASET_STATS:-./runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/dataset_stats.json}
export EXTRA_MANAGER_ARGS=${EXTRA_MANAGER_ARGS:-+EVALUATION.policy_subprocess=true}

SELECT_GROUPS=${SELECT_GROUPS:-a1_step070,control,residual_only,residual_clip025}
A1_GPUS=${A1_GPUS:-0}
CONTROL_GPUS=${CONTROL_GPUS:-1}
RESIDUAL_ONLY_GPUS=${RESIDUAL_ONLY_GPUS:-2}
RESIDUAL_CLIP_GPUS=${RESIDUAL_CLIP_GPUS:-3}
RUN_DATE=${RUN_DATE:-20260605}

should_run() {
  case ",$SELECT_GROUPS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

start_eval() {
  local label=$1
  local task_config=$2
  local ckpt=$3
  local gpus=$4

  if ! should_run "$label"; then
    echo "Skipping group: $label"
    return 0
  fi

  export TASK_CONFIG="$task_config"
  export CKPT="$ckpt"
  export GPUS="$gpus"
  export RUN_TAG="a1_${label}_a2_attribution_probe_v1_steps${NUM_INFERENCE_STEPS}_${NUM_TRIALS}trials_${RUN_DATE}"
  export OUTPUT_DIR="./evaluate_results/libero/${RUN_TAG}"
  export MANAGER_SESSION_NAME="libero_${label}_a2_attr_manager_${RUN_DATE}"
  export WORKER_SESSION_NAME="libero_${label}_a2_attr_workers_${RUN_DATE}"

  bash experiments/libero/run_libero_task_set_eval.sh
}

start_eval \
  a1_step070 \
  libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5 \
  ./runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/checkpoints/weights/step_070000.pt \
  "$A1_GPUS"

start_eval \
  control \
  libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5 \
  ./runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_continue_control_weights_from070k_10k_20260603/checkpoints/weights/step_010000.pt \
  "$CONTROL_GPUS"

start_eval \
  residual_only \
  libero_one_step_meanflow_a1_residual_only_lora_eqanchor_2cam224_5e-5 \
  ./runs/libero_one_step_meanflow_a1_residual_only_lora_eqanchor_2cam224_5e-5/a1_2_residual_only_weights_from070k_10k_20260604/checkpoints/weights/step_010000.pt \
  "$RESIDUAL_ONLY_GPUS"

start_eval \
  residual_clip025 \
  libero_one_step_meanflow_a1_residual_clip_lora_eqanchor_2cam224_5e-5 \
  ./runs/libero_one_step_meanflow_a1_residual_clip_lora_eqanchor_2cam224_5e-5/a1_3_residual_clip025_weights_from070k_10k_20260604/checkpoints/weights/step_010000.pt \
  "$RESIDUAL_CLIP_GPUS"
