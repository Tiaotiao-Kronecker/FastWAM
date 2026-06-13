#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.conda/fastwam/bin:${PATH}"
export FASTWAM_GPU_IDS="${FASTWAM_GPU_IDS:-0,1,4,7}"
export CUDA_VISIBLE_DEVICES="${FASTWAM_GPU_IDS}"
export RUN_ID="${RUN_ID:-a1_4_full_mix_from070k_to80000_h2002_4gpu_20260613}"

A1_STATE="./runs/libero_one_step_meanflow_a1_lora_eqanchor_2cam224_5e-5/a1_lora_eqanchor_sync_20260529_174000/checkpoints/state/step_070000"
DATA_ROOT="/DATA/disk0/shared/datasets/libero_mujoco3.3.2"
TEXT_CACHE="/DATA/disk0/shared/datasets/text_embeds_cache/libero"

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT:-29500}" \
  --gpu_ids "${FASTWAM_GPU_IDS}" \
  scripts/train.py \
  "output_dir=./runs/libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5/${RUN_ID}" \
  task=libero_one_step_meanflow_a1_mix_lora_eqanchor_2cam224_5e-5 \
  max_steps=80000 \
  save_every=5000 \
  save_training_state=true \
  eval_every=0 \
  log_every=10 \
  resume="${A1_STATE}" \
  "data.train.dataset_dirs=[${DATA_ROOT}/libero_spatial_no_noops_lerobot,${DATA_ROOT}/libero_object_no_noops_lerobot,${DATA_ROOT}/libero_goal_no_noops_lerobot,${DATA_ROOT}/libero_10_no_noops_lerobot]" \
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}" \
  wandb.enabled=false \
  wandb.name=a1_4_full_mix_from070k_to80000_h2002_4gpu \
  wandb.group=meanflow-a1-full-continue
