#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: $0 <gpu> <task_id> <instruction_tag> <instruction> <pair_prefix> <seed_csv> [suite] [image_shift_preset]" >&2
  exit 1
fi

GPU="$1"
TASK_ID="$2"
TAG="$3"
INSTRUCTION="$4"
PAIR_PREFIX="$5"
SEED_CSV="$6"
SUITE="${7:-libero_object}"
IMAGE_SHIFT_PRESET="${8:-clean}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${BAS_PYTHON_BIN:-python3}"
RUNNER="${BAS_OPENPI_RUNNER:-$SCRIPT_DIR/eval_openpi_libero.py}"
OUTPUT_ROOT="${BAS_OUTPUT_ROOT:-$REPO_ROOT/runs/openpi_main_campaign}"

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
    --output-dir "$OUTPUT_ROOT" \
    --pair-id "${PAIR_PREFIX}_seed${SEED}" \
    --instruction-tag "$TAG" \
    --instruction-override "$INSTRUCTION" \
    --image-shift-preset "$IMAGE_SHIFT_PRESET" \
    --run-note "openpi campaign task=${TASK_ID} tag=${TAG} seed=${SEED}"
done
