#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.analysis.latency import summarize_latency

def main():
    parser = argparse.ArgumentParser(description="Chunk latency percentiles and 200-chunk x 3-pass completeness checks.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-chunks", type=int, default=200)
    parser.add_argument("--min-passes", type=int, default=3)
    parser.add_argument("--allow-unsynchronized", action="store_true", help="Describe wall-clock traces; protocol_complete remains false.")
    args = parser.parse_args()
    result = summarize_latency(json.loads(args.input.read_text()), min_chunks=args.min_chunks, min_passes=args.min_passes, allow_unsynchronized=args.allow_unsynchronized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
