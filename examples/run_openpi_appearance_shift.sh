#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$REPO_ROOT/scripts/eval_openpi_appearance_libero.py" \
  --suite libero_object \
  --task-ids 9 \
  --num-trials-per-task 2 \
  --seed 7 \
  --shift-preset bg_dark_dim \
  --output-dir "$REPO_ROOT/runs/example_openpi_appearance" \
  --instruction-override "pick up the orange juice and place it in the basket"
