#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$REPO_ROOT/scripts/eval_openpi_libero.py" \
  --suite libero_object \
  --task-ids 7 \
  --num-trials-per-task 2 \
  --seed 7 \
  --output-dir "$REPO_ROOT/runs/example_openpi_semantic_break" \
  --pair-id "example_task7_seed7" \
  --instruction-tag clean \
  --instruction-override "pick up the milk and place it in the basket"
