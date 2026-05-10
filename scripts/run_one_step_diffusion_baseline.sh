#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${1:?Usage: CKPT=<path> [STATS=<path>] bash scripts/run_one_step_diffusion_baseline.sh <robotwin|libero> [hydra_overrides...]}"
shift

CKPT="${CKPT:?Set CKPT to the checkpoint path.}"
STATS="${STATS:-}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
STEPS="${STEPS:-1 2 4 10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS=("$@")

if [[ ! -f "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [[ -n "${STATS}" && ! -f "${STATS}" ]]; then
  echo "Error: dataset stats not found: ${STATS}" >&2
  exit 1
fi

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_GPUS}" || (( NUM_GPUS <= 0 )); then
  echo "Error: NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 1
fi

if ! is_integer "${MAX_TASKS_PER_GPU}" || (( MAX_TASKS_PER_GPU <= 0 )); then
  echo "Error: MAX_TASKS_PER_GPU must be a positive integer, got: ${MAX_TASKS_PER_GPU}" >&2
  exit 1
fi

case "${BENCHMARK}" in
  robotwin)
    TASK="${TASK:-robotwin_uncond_3cam_384_1e-4}"
    ENTRY=(python experiments/robotwin/run_robotwin_manager.py)
    if [[ "${LINK_ROBOTWIN_POLICY:-1}" == "1" ]]; then
      mkdir -p third_party/RoboTwin/policy
      ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
    fi
    ;;
  libero)
    TASK="${TASK:-libero_uncond_2cam224_1e-4}"
    ENTRY=(python experiments/libero/run_libero_manager.py)
    ;;
  *)
    echo "Error: BENCHMARK must be one of: robotwin, libero. Got: ${BENCHMARK}" >&2
    exit 1
    ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-./evaluate_results/one_step_diffusion_baseline/${BENCHMARK}/${RUN_ID}}"
mkdir -p "${OUTPUT_ROOT}"

echo "[baseline] benchmark=${BENCHMARK}"
echo "[baseline] task=${TASK}"
echo "[baseline] ckpt=${CKPT}"
echo "[baseline] stats=${STATS:-<none>}"
echo "[baseline] steps=${STEPS}"
echo "[baseline] output_root=${OUTPUT_ROOT}"
echo "[baseline] num_gpus=${NUM_GPUS} max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"

for step in ${STEPS}; do
  if ! is_integer "${step}" || (( step <= 0 )); then
    echo "Error: each value in STEPS must be a positive integer, got: ${step}" >&2
    exit 1
  fi

  run_dir="${OUTPUT_ROOT}/${RUN_ID}_steps_${step}"
  cmd=(
    "${ENTRY[@]}"
    "task=${TASK}"
    "ckpt=${CKPT}"
    "EVALUATION.num_inference_steps=${step}"
    "EVALUATION.output_dir=${run_dir}"
    "MULTIRUN.num_gpus=${NUM_GPUS}"
    "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
  )

  if [[ -n "${STATS}" ]]; then
    cmd+=("EVALUATION.dataset_stats_path=${STATS}")
  fi

  cmd+=("${EXTRA_ARGS[@]}")

  echo "[baseline] running steps=${step}"
  printf '[baseline] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done

echo "[baseline] done: ${OUTPUT_ROOT}"
