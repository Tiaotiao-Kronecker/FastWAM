#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

GPUS=${GPUS:-0,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
NUM_TRIALS=${NUM_TRIALS:-50}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}
POLL_SECONDS=${POLL_SECONDS:-60}

SHORTCUT_SESSION=${SHORTCUT_SESSION:-libero_shortcut_pilot2k}
MEANFLOW_SESSION=${MEANFLOW_SESSION:-libero_meanflow_pilot2k}

SHORTCUT_CKPT=${SHORTCUT_CKPT:-./runs/libero_one_step_shortcut_pilot2k_20260512/checkpoints/weights/step_002000.pt}
MEANFLOW_CKPT=${MEANFLOW_CKPT:-./runs/libero_one_step_meanflow_pilot2k_20260512/checkpoints/weights/step_002000.pt}

SHORTCUT_OUTPUT=${SHORTCUT_OUTPUT:-./evaluate_results/libero/shortcut_1_pilot2k_20260512}
MEANFLOW_OUTPUT=${MEANFLOW_OUTPUT:-./evaluate_results/libero/meanflow_1_pilot2k_20260512}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

count_results() {
    local output_dir=$1
    if [ ! -d "$output_dir" ]; then
        echo 0
        return
    fi
    find "$output_dir" -type f -name "gpu*_task*_results.json" | wc -l
}

wait_training_done() {
    local method=$1
    local session=$2
    local ckpt=$3

    while true; do
        if [ -f "$ckpt" ] && ! tmux has-session -t "$session" 2>/dev/null; then
            log "$method training complete: $ckpt"
            return
        fi

        if ! tmux has-session -t "$session" 2>/dev/null && [ ! -f "$ckpt" ]; then
            log "$method training session ended but checkpoint is missing: $ckpt"
            exit 2
        fi

        log "$method training running; waiting for $ckpt"
        sleep "$POLL_SECONDS"
    done
}

start_eval() {
    local method=$1
    local task_config=$2
    local ckpt=$3
    local output_dir=$4
    local manager_session="${method}_eval_manager"
    local worker_session="${method}_eval_workers"

    local result_count
    result_count=$(count_results "$output_dir")
    if [ "$result_count" -ge 40 ] && [ -f "$output_dir/summary.csv" ]; then
        log "$method eval already complete: $output_dir"
        return
    fi

    if tmux has-session -t "$manager_session" 2>/dev/null; then
        log "$method eval manager already running: $manager_session"
        return
    fi

    log "Starting $method eval: output=$output_dir checkpoint=$ckpt"
    TASK_CONFIG="$task_config" \
    CKPT="$ckpt" \
    OUTPUT_DIR="$output_dir" \
    RUN_TAG="$method" \
    NUM_INFERENCE_STEPS=1 \
    NUM_TRIALS="$NUM_TRIALS" \
    SAVE_ROLLOUT_VIDEO="$SAVE_ROLLOUT_VIDEO" \
    GPUS="$GPUS" \
    WORKERS_PER_GPU="$WORKERS_PER_GPU" \
    MANAGER_SESSION_NAME="$manager_session" \
    WORKER_SESSION_NAME="$worker_session" \
    experiments/libero/run_libero_multiworker_batch.sh
}

wait_eval_done() {
    local method=$1
    local output_dir=$2
    local manager_session="${method}_eval_manager"

    while true; do
        local result_count
        result_count=$(count_results "$output_dir")

        if [ -s "$output_dir/failed_tasks.txt" ]; then
            log "$method eval has failures. See $output_dir/failed_tasks.txt"
            exit 3
        fi

        if [ "$result_count" -ge 40 ] && [ -f "$output_dir/summary.csv" ]; then
            log "$method eval complete: $output_dir"
            return
        fi

        if ! tmux has-session -t "$manager_session" 2>/dev/null; then
            log "$method eval manager stopped before completion; restarting/resuming."
            if [ "$method" = "shortcut_1_pilot2k_20260512" ]; then
                start_eval "$method" "libero_one_step_shortcut_2cam224_1e-4" "$SHORTCUT_CKPT" "$SHORTCUT_OUTPUT"
            else
                start_eval "$method" "libero_one_step_meanflow_2cam224_1e-4" "$MEANFLOW_CKPT" "$MEANFLOW_OUTPUT"
            fi
        else
            log "$method eval running: results=$result_count/40"
        fi

        sleep "$POLL_SECONDS"
    done
}

wait_training_done "shortcut_1_pilot2k_20260512" "$SHORTCUT_SESSION" "$SHORTCUT_CKPT"
wait_training_done "meanflow_1_pilot2k_20260512" "$MEANFLOW_SESSION" "$MEANFLOW_CKPT"

start_eval "shortcut_1_pilot2k_20260512" "libero_one_step_shortcut_2cam224_1e-4" "$SHORTCUT_CKPT" "$SHORTCUT_OUTPUT"
wait_eval_done "shortcut_1_pilot2k_20260512" "$SHORTCUT_OUTPUT"

start_eval "meanflow_1_pilot2k_20260512" "libero_one_step_meanflow_2cam224_1e-4" "$MEANFLOW_CKPT" "$MEANFLOW_OUTPUT"
wait_eval_done "meanflow_1_pilot2k_20260512" "$MEANFLOW_OUTPUT"

log "One-step pilot eval chain complete."
cat "$SHORTCUT_OUTPUT/summary.csv"
cat "$MEANFLOW_OUTPUT/summary.csv"
