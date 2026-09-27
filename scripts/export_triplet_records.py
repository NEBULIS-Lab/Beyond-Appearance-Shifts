#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.records.cached_records import export_triplet_records

def main():
    parser = argparse.ArgumentParser(description="Validate and package precomputed matched triplets; no carrier inference.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_triplet_records(args.input, args.output, jsonl=args.jsonl), indent=2))

if __name__ == "__main__":
    main()
