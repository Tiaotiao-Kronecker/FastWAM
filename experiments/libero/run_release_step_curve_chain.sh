#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

STEPS=${STEPS:-"4 2 1"}
GPUS=${GPUS:-0,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}
POLL_SECONDS=${POLL_SECONDS:-60}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-./evaluate_results/libero/release_baseline_20260512_steps}

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

has_failures() {
    local output_dir=$1
    local failed_file="$output_dir/failed_tasks.txt"
    [ -s "$failed_file" ]
}

start_step_if_needed() {
    local step=$1
    local output_dir="${OUTPUT_PREFIX}${step}"
    local manager_session="release${step}_steps${step}_manager"
    local worker_session="release${step}_steps${step}_workers"
    local result_count
    result_count=$(count_results "$output_dir")

    if [ "$result_count" -ge 40 ] && [ -f "$output_dir/summary.csv" ]; then
        log "release_${step} already complete: $output_dir"
        return
    fi

    if tmux has-session -t "$manager_session" 2>/dev/null; then
        log "release_${step} manager already running: $manager_session"
        return
    fi

    log "Starting release_${step}: output=$output_dir"
    RUN_TAG="release_${step}_steps${step}" \
    OUTPUT_DIR="$output_dir" \
    NUM_INFERENCE_STEPS="$step" \
    WORKERS_PER_GPU="$WORKERS_PER_GPU" \
    SAVE_ROLLOUT_VIDEO="$SAVE_ROLLOUT_VIDEO" \
    MANAGER_SESSION_NAME="$manager_session" \
    WORKER_SESSION_NAME="$worker_session" \
    GPUS="$GPUS" \
    experiments/libero/run_libero_multiworker_batch.sh
}

wait_step_done() {
    local step=$1
    local output_dir="${OUTPUT_PREFIX}${step}"
    local manager_session="release${step}_steps${step}_manager"

    while true; do
        local result_count
        result_count=$(count_results "$output_dir")

        if has_failures "$output_dir"; then
            log "release_${step} has failures; stopping chain. See $output_dir/failed_tasks.txt"
            exit 2
        fi

        if [ "$result_count" -ge 40 ] && [ -f "$output_dir/summary.csv" ]; then
            log "release_${step} complete: $output_dir"
            return
        fi

        if ! tmux has-session -t "$manager_session" 2>/dev/null; then
            log "release_${step} manager is not running and output is incomplete; restarting/resuming."
            start_step_if_needed "$step"
        else
            log "release_${step} running: results=$result_count/40"
        fi

        sleep "$POLL_SECONDS"
    done
}

for step in $STEPS; do
    start_step_if_needed "$step"
    wait_step_done "$step"
done

log "Release step curve complete for steps: $STEPS"
