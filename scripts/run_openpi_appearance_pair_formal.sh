#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: $0 <gpu> <task_id> <instruction> <pair_prefix> <seed_csv> <shift_preset> [suite]" >&2
  exit 1
fi

GPU="$1"
TASK_ID="$2"
INSTRUCTION="$3"
PAIR_PREFIX="$4"
SEED_CSV="$5"
SHIFT_PRESET="$6"
SUITE="${7:-libero_object}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${BAS_PYTHON_BIN:-python3}"
RUNNER="${BAS_OPENPI_APPEARANCE_RUNNER:-$SCRIPT_DIR/eval_openpi_appearance_libero.py}"
OUTPUT_ROOT="${BAS_OUTPUT_ROOT:-$REPO_ROOT/runs/openpi_appearance_formal}"

# Preserve the device visibility inherited from the execution environment.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_ROOT"

IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
for SEED in "${SEEDS[@]}"; do
  "$PYTHON_BIN" "$RUNNER" \
    --suite "$SUITE" \
    --task-ids "$TASK_ID" \
    --num-trials-per-task 50 \
    --num-steps-wait 10 \
    --max-steps-override 180 \
    --replan-steps 5 \
    --seed "$SEED" \
    --shift-preset clean \
    --instruction-override "$INSTRUCTION" \
    --output-dir "$OUTPUT_ROOT" \
    --run-note "openpi appearance formal task=${TASK_ID} shift=clean seed=${SEED}"
done
for SEED in "${SEEDS[@]}"; do
  "$PYTHON_BIN" "$RUNNER" \
    --suite "$SUITE" \
    --task-ids "$TASK_ID" \
    --num-trials-per-task 50 \
    --num-steps-wait 10 \
    --max-steps-override 180 \
    --replan-steps 5 \
    --seed "$SEED" \
    --shift-preset "$SHIFT_PRESET" \
    --instruction-override "$INSTRUCTION" \
    --output-dir "$OUTPUT_ROOT" \
    --run-note "openpi appearance formal task=${TASK_ID} shift=${SHIFT_PRESET} seed=${SEED}"
done
