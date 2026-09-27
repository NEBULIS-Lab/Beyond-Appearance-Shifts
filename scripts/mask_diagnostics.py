#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from bas_vla.analysis.masks import corrupt_mask, summarize_mask_outcomes

def main():
    parser = argparse.ArgumentParser(description="Offline binary-mask corruption or paired rollout-outcome aggregation.")
    commands = parser.add_subparsers(dest="command", required=True)
    masks = commands.add_parser("corrupt")
    masks.add_argument("--input", required=True, type=Path, help="2D binary .npy reference mask")
    masks.add_argument("--target-mask", type=Path)
    masks.add_argument("--output", required=True, type=Path, help="Output .npy path; sidecar .json stores achieved IoU")
    masks.add_argument("--kind", choices=("normal", "iou", "missed_target", "random"), required=True)
    masks.add_argument("--target-iou", type=float)
    masks.add_argument("--seed", type=int, required=True)
    outcomes = commands.add_parser("summarize")
    outcomes.add_argument("--input", required=True, type=Path)
    outcomes.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "corrupt":
        if args.output.suffix != ".npy":
            parser.error("mask output must end in .npy")
        target = np.load(args.target_mask, allow_pickle=False) if args.target_mask else None
        mask, metadata = corrupt_mask(np.load(args.input, allow_pickle=False), kind=args.kind, seed=args.seed, target_iou=args.target_iou, target_mask=target)
        np.save(args.output, mask, allow_pickle=False)
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        result = summarize_mask_outcomes(json.loads(args.input.read_text()))
        args.output.write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
