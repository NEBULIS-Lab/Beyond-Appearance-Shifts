#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${BAS_OPENPI_FORMAL_WORKER:-$SCRIPT_DIR/run_openpi_formal_worker.sh}"
MATRIX="${1:-$SCRIPT_DIR/../configs/openpi/main_campaign.tsv}"
LOGDIR="${BAS_OUTPUT_ROOT:-$SCRIPT_DIR/../runs/openpi_formal}/logs"

if [ ! -f "$MATRIX" ]; then
  echo "matrix file not found: $MATRIX" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

while IFS=$'\t' read -r GPU TASK_ID TAG INSTRUCTION PAIR_PREFIX SEEDS; do
  if [ -z "${GPU:-}" ] || [ "${GPU:0:1}" = "#" ]; then
    continue
  fi

  SAFE_PREFIX="$(printf '%s' "$PAIR_PREFIX" | tr '/ ' '__')"
  LOG_PATH="$LOGDIR/${SAFE_PREFIX}_${TAG}.log"
  PID_PATH="$LOGDIR/${SAFE_PREFIX}_${TAG}.pid"

  nohup bash "$WORKER" "$GPU" "$TASK_ID" "$TAG" "$INSTRUCTION" "$PAIR_PREFIX" "$SEEDS" \
    > "$LOG_PATH" 2>&1 < /dev/null &
  echo $! > "$PID_PATH"
  echo "launched $PAIR_PREFIX $TAG on gpu=$GPU pid=$(cat "$PID_PATH")"
done < "$MATRIX"
