#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$REPO_ROOT/scripts/eval_openvla_oft_libero.py" \
  --pairs-config "$REPO_ROOT/configs/openvla_oft/semantic_break_pairs_main.json" \
  --pair-id break_target_object \
  --num-trials 1 \
  --seed 7 \
  --output-dir "$REPO_ROOT/runs/example_openvla_oft_semantic_break"
