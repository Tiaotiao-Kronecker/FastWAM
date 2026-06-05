#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

BRANCH=${BRANCH:-mix}
MAX_STEPS=${MAX_STEPS:-300}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
SAVE_EVERY=${SAVE_EVERY:-$MAX_STEPS}
WANDB_ENABLED=${WANDB_ENABLED:-false}
MASTER_PORT=${MASTER_PORT:-29610}
export MASTER_PORT

A1_70K_CKPT=${A1_70K_CKPT:-./runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/checkpoints/weights/step_070000.pt}

case "$BRANCH" in
  mix)
    TASK_CONFIG=libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5
    RUN_PREFIX=a1_4_mix
    ;;
  mix_clip|mix+clip)
    TASK_CONFIG=libero_one_step_meanflow_a1_mix_clip_lora_eqanchor_2cam224_5e-5
    RUN_PREFIX=a1_4_mix_clip025
    ;;
  *)
    echo "Unknown BRANCH: $BRANCH. Expected 'mix' or 'mix_clip'." >&2
    exit 1
    ;;
esac

RUN_ID=${RUN_ID:-${RUN_PREFIX}_from070k_${MAX_STEPS}steps_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-./runs/${TASK_CONFIG}/${RUN_ID}}

echo "Branch: $BRANCH"
echo "Task config: $TASK_CONFIG"
echo "A1 70k checkpoint: $A1_70K_CKPT"
echo "Output dir: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

bash scripts/train_zero2.sh "$NPROC_PER_NODE" \
  task="$TASK_CONFIG" \
  "resume=$A1_70K_CKPT" \
  "max_steps=$MAX_STEPS" \
  "save_every=$SAVE_EVERY" \
  "output_dir=$OUTPUT_DIR" \
  "wandb.enabled=$WANDB_ENABLED" \
  "wandb.name=$RUN_ID" \
  "$@"
