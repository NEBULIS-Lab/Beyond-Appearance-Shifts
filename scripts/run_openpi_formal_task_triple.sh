#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 7 ]; then
  echo "usage: $0 <gpu> <task_id> <clean_instruction> <control_instruction> <break_instruction> <pair_prefix> <seed_csv> [image_shift_preset] [suite]" >&2
  exit 1
fi

GPU="$1"
TASK_ID="$2"
CLEAN_INSTRUCTION="$3"
CONTROL_INSTRUCTION="$4"
BREAK_INSTRUCTION="$5"
PAIR_PREFIX="$6"
SEED_CSV="$7"
IMAGE_SHIFT_PRESET="${8:-clean}"
SUITE="${9:-libero_object}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${BAS_OPENPI_FORMAL_WORKER:-$SCRIPT_DIR/run_openpi_formal_worker.sh}"

bash "$WORKER" "$GPU" "$TASK_ID" clean "$CLEAN_INSTRUCTION" "$PAIR_PREFIX" "$SEED_CSV" "$IMAGE_SHIFT_PRESET" "$SUITE"
bash "$WORKER" "$GPU" "$TASK_ID" control "$CONTROL_INSTRUCTION" "$PAIR_PREFIX" "$SEED_CSV" "$IMAGE_SHIFT_PRESET" "$SUITE"
bash "$WORKER" "$GPU" "$TASK_ID" break "$BREAK_INSTRUCTION" "$PAIR_PREFIX" "$SEED_CSV" "$IMAGE_SHIFT_PRESET" "$SUITE"
