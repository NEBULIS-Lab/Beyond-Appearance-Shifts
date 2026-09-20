#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Supply matched carrier triplets and their exact action convention.
: "${RECORDS_PATH:?Set RECORDS_PATH to your precomputed JSON records}"
: "${CARRIER_ID:?Set CARRIER_ID to the frozen carrier checkpoint identifier}"
: "${ACTION_SPACE:?Set ACTION_SPACE to the runtime action convention}"
: "${ACTION_NORMALIZATION:?Set ACTION_NORMALIZATION to the action convention identifier}"
python3 "$REPO_ROOT/scripts/train_breaking_adapter.py" \
  --records-path "$RECORDS_PATH" \
  --action-space "$ACTION_SPACE" --action-layout "${ACTION_LAYOUT:-per_step}" \
  --carrier "$CARRIER_ID" --action-normalization "$ACTION_NORMALIZATION" \
  --config "$REPO_ROOT/configs/training/paper_three_component.json" \
  --device "${DEVICE:-cpu}" \
  --output-dir "${OUTPUT_DIR:-$REPO_ROOT/runs/breaking_adapter}"
