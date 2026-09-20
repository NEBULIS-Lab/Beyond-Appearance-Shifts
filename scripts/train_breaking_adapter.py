#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.breaking.training import (
    OBJECTIVE_PRESETS,
    training_config,
    attach_bow_features,
    build_vocab_from_records,
    save_training_artifacts,
    split_records,
    train_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BAS-VLA breaking residual adapter from precomputed action records."
    )
    parser.add_argument("--records-path", type=Path, required=True, help="Path to a JSON list of cached records.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for model artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", help="Explicit compute device; parsing never probes accelerators.")
    parser.add_argument("--objective-preset", choices=OBJECTIVE_PRESETS, default=None)
    parser.add_argument("--config", type=Path, help="JSON with an objective_preset and optional overrides.")
    parser.add_argument("--action-space", required=True, help="Runtime action convention, e.g. libero_env or oft_pre_process_action.")
    parser.add_argument("--action-layout", choices=("per_step", "flattened_chunk"), default="per_step")
    parser.add_argument("--carrier", required=True, help="Exact frozen carrier/checkpoint identifier.")
    parser.add_argument("--action-normalization", required=True, help="Exact action coordinates, units and normalization identifier.")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--delta-scale", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--lambda-consistency", type=float, default=None)
    parser.add_argument("--lambda-margin", type=float, default=None)
    parser.add_argument("--lambda-anchor", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lambda-imitation", type=float)
    parser.add_argument("--lambda-expert-margin", type=float)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("records-path must point to a JSON list of record dictionaries")
    return payload


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    from bas_vla.records.cached_records import validate_cached_records
    records = load_records(args.records_path)
    validate_cached_records(records)
    for key in ("action_space", "action_layout", "action_normalization"):
        if any(key in record and record[key] != getattr(args, key) for record in records):
            raise ValueError(f"record {key} disagrees with declared training convention")
    vocab = build_vocab_from_records([record for record in records if record["split"] == "train"])
    records = attach_bow_features(records, vocab)
    by_split = split_records(records)
    train_records = by_split.get("train", [])
    val_records = by_split.get("val", [])

    if not train_records or not val_records:
        raise RuntimeError("records must contain both 'train' and 'val' splits")

    payload = json.loads(args.config.read_text()) if args.config else {}
    preset = args.objective_preset or payload.get("objective_preset", "legacy")
    overrides = dict(payload.get("overrides", {}))
    for key in ("hidden_dim", "delta_scale", "margin", "lambda_consistency", "lambda_margin",
                "lambda_anchor", "lambda_imitation", "lambda_expert_margin", "lr", "epochs", "batch_size"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    cfg = training_config(preset, **overrides)

    result = train_adapter(
        train_records=train_records,
        val_records=val_records,
        cfg=cfg,
        device=torch.device(args.device),
    )
    save_training_artifacts(args.output_dir, result, vocab, cfg, provenance={
        "carrier": args.carrier, "action_normalization": args.action_normalization,
        "action_space": args.action_space, "action_layout": args.action_layout,
        "seed": args.seed, "record_count": len(records),
        "records_sha256": hashlib.sha256(args.records_path.read_bytes()).hexdigest(),
    })
    print(f"[bas-vla] saved artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
