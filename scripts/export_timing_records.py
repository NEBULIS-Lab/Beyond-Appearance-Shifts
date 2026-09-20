#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.records.diagnostics import timing_rows_from_episodes


def main():
    parser=argparse.ArgumentParser(description="Flatten recorded episode traces; never invent synchronization or warm-up status.")
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--jsonl",action="store_true")
    parser.add_argument("--pass-id",help="Explicit timing pass if not present in traces")
    args=parser.parse_args()
    text=args.input.read_text()
    payload=[json.loads(line) for line in text.splitlines() if line.strip()] if args.jsonl else json.loads(text)
    rows=timing_rows_from_episodes(payload,pass_id=args.pass_id)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":main()
