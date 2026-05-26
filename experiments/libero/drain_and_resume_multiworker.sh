#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/DATA/disk2/wangchen/projects/FastWAM}
cd "$ROOT_DIR"

OUTPUT_DIR=${OUTPUT_DIR:-./evaluate_results/libero/release_baseline_20260512_steps10_v2}
OLD_MANAGER_SESSION=${OLD_MANAGER_SESSION:-libero_release10_20260512_v2}
OLD_WORKER_SESSION=${OLD_WORKER_SESSION:-libero_test_v3}
NEW_RUN_TAG=${NEW_RUN_TAG:-release10_v2_multi_resume}
NEW_MANAGER_SESSION=${NEW_MANAGER_SESSION:-"${NEW_RUN_TAG}_manager"}
NEW_WORKER_SESSION=${NEW_WORKER_SESSION:-"${NEW_RUN_TAG}_workers"}
GPUS=${GPUS:-0,2,3}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-false}
POLL_SECONDS=${POLL_SECONDS:-10}
ABORT_ON_FAILURE=${ABORT_ON_FAILURE:-true}

PENDING_TASKS_FILE="$OUTPUT_DIR/pending_tasks.txt"
TASK_GPU_MAP_FILE="$OUTPUT_DIR/task_gpu_map.txt"
FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"
SWITCH_LOG_DIR="$OUTPUT_DIR/switch_logs"
mkdir -p "$SWITCH_LOG_DIR"
SWITCH_TS=$(date +%Y%m%d_%H%M%S)
BACKUP_PENDING_FILE="$SWITCH_LOG_DIR/pending_tasks.before_multiworker_switch.${SWITCH_TS}.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [ ! -d "$OUTPUT_DIR" ]; then
    log "Output directory not found: $OUTPUT_DIR"
    exit 1
fi

if [ -f "$PENDING_TASKS_FILE" ]; then
    cp "$PENDING_TASKS_FILE" "$BACKUP_PENDING_FILE"
    : > "$PENDING_TASKS_FILE"
    log "Backed up and drained pending queue: $BACKUP_PENDING_FILE"
else
    log "Pending queue not found; continuing: $PENDING_TASKS_FILE"
fi

log "Waiting for currently running tasks to finish before switching."
while true; do
    running_count=0
    if [ -f "$TASK_GPU_MAP_FILE" ]; then
        running_count=$(grep -cve '^[[:space:]]*$' "$TASK_GPU_MAP_FILE" || true)
    fi

    failed_count=0
    if [ -f "$FAILED_TASKS_FILE" ]; then
        failed_count=$(grep -cve '^[[:space:]]*$' "$FAILED_TASKS_FILE" || true)
    fi

    result_count=$(find "$OUTPUT_DIR" -type f -name "gpu*_task*_results.json" | wc -l)
    log "running=${running_count}, completed_results=${result_count}, failed=${failed_count}"

    if [ "$failed_count" -gt 0 ] && [ "$ABORT_ON_FAILURE" = "true" ]; then
        log "Detected failed tasks; not starting multiworker resume. See $FAILED_TASKS_FILE"
        exit 2
    fi

    if [ "$running_count" -eq 0 ]; then
        break
    fi

    sleep "$POLL_SECONDS"
done

log "Current tasks drained. Stopping old tmux sessions if present."
tmux kill-session -t "$OLD_MANAGER_SESSION" 2>/dev/null || true
tmux kill-session -t "$OLD_WORKER_SESSION" 2>/dev/null || true

log "Starting multiworker resume."
OUTPUT_DIR="$OUTPUT_DIR" \
RUN_TAG="$NEW_RUN_TAG" \
MANAGER_SESSION_NAME="$NEW_MANAGER_SESSION" \
WORKER_SESSION_NAME="$NEW_WORKER_SESSION" \
GPUS="$GPUS" \
WORKERS_PER_GPU="$WORKERS_PER_GPU" \
SAVE_ROLLOUT_VIDEO="$SAVE_ROLLOUT_VIDEO" \
experiments/libero/run_libero_multiworker_batch.sh

log "Switch complete."
log "New manager session: $NEW_MANAGER_SESSION"
log "New worker session: $NEW_WORKER_SESSION"
