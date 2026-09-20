#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <tsv_path> [suite]" >&2
  exit 1
fi

TSV_PATH="$1"
SUITE="${2:-libero_object}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIPLE="${BAS_OPENPI_APPEARANCE_FORMAL_RUNNER:-$SCRIPT_DIR/run_openpi_appearance_pair_formal.sh}"

tail -n +2 "$TSV_PATH" | while IFS=$'\t' read -r GPU TASK_ID INSTRUCTION SHIFT_PRESET PAIR_PREFIX SEEDS; do
  if [ -z "${GPU:-}" ]; then
    continue
  fi
  bash "$TRIPLE" "$GPU" "$TASK_ID" "$INSTRUCTION" "$PAIR_PREFIX" "$SEEDS" "$SHIFT_PRESET" "$SUITE"
done
