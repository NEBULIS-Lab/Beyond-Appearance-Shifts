#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: $0 <gpu> <task_id> <instruction> <pair_prefix> <num_trials> <shift_preset> [seed] [suite]" >&2
  exit 1
fi

GPU="$1"
TASK_ID="$2"
INSTRUCTION="$3"
PAIR_PREFIX="$4"
NUM_TRIALS="$5"
SHIFT_PRESET="$6"
SEED="${7:-7}"
SUITE="${8:-libero_object}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${BAS_PYTHON_BIN:-python3}"
RUNNER="${BAS_OPENPI_APPEARANCE_RUNNER:-$SCRIPT_DIR/eval_openpi_appearance_libero.py}"
OUTPUT_ROOT="${BAS_OUTPUT_ROOT:-$REPO_ROOT/runs/openpi_appearance_pairs}"

# Preserve the device visibility inherited from the execution environment.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_ROOT"

run_condition() {
  local shift="$1"
  "$PYTHON_BIN" "$RUNNER" \
    --suite "$SUITE" \
    --task-ids "$TASK_ID" \
    --num-trials-per-task "$NUM_TRIALS" \
    --num-steps-wait 10 \
    --max-steps-override 180 \
    --replan-steps 5 \
    --seed "$SEED" \
    --shift-preset "$shift" \
    --instruction-override "$INSTRUCTION" \
    --output-dir "$OUTPUT_ROOT" \
    --run-note "openpi appearance pair task=${TASK_ID} shift=${shift} seed=${SEED}"
}

run_condition clean
run_condition "$SHIFT_PRESET"
