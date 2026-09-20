#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.analysis.gate import summarize_gate

def main():
    parser = argparse.ArgumentParser(description="2x2 gate matrix from explicit reference actions and labels.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--unit", choices=("chunk", "episode"), default="chunk")
    args = parser.parse_args()
    result = summarize_gate(json.loads(args.input.read_text()), activation_threshold=args.activation_threshold, unit=args.unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
