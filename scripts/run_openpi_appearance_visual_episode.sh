#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 8 ]; then
  echo "usage: $0 <gpu> <task_id> <seed> <episode_idx> <shift_preset> <instruction> <pair_prefix> <tag> [suite]" >&2
  exit 1
fi

GPU="$1"
TASK_ID="$2"
SEED="$3"
EPISODE_IDX="$4"
SHIFT_PRESET="$5"
INSTRUCTION="$6"
PAIR_PREFIX="$7"
TAG="$8"
SUITE="${9:-libero_object}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${BAS_PYTHON_BIN:-python3}"
RUNNER="${BAS_OPENPI_APPEARANCE_RUNNER:-$SCRIPT_DIR/eval_openpi_appearance_libero.py}"
OUTPUT_ROOT="${BAS_OUTPUT_ROOT:-$REPO_ROOT/runs/openpi_appearance_visuals}"

# Preserve the device visibility inherited from the execution environment.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" "$RUNNER" \
  --suite "$SUITE" \
  --task-ids "$TASK_ID" \
  --num-trials-per-task 1 \
  --episode-indices "$EPISODE_IDX" \
  --num-steps-wait 10 \
  --max-steps-override 180 \
  --replan-steps 5 \
  --seed "$SEED" \
  --shift-preset "$SHIFT_PRESET" \
  --instruction-override "$INSTRUCTION" \
  --output-dir "$OUTPUT_ROOT" \
  --save-rollout-frames-limit 20 \
  --save-rollout-frame-stride 8 \
  --save-rollout-media \
  --run-note "openpi appearance visual task=${TASK_ID} shift=${SHIFT_PRESET} seed=${SEED} ep=${EPISODE_IDX} tag=${TAG}"
