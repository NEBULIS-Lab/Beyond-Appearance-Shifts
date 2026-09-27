#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.analysis.wilson import count_rows, count_table

def main():
    parser = argparse.ArgumentParser(description="Wilson intervals from a JSON list of label/successes/total rows.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--format", choices=("json", "markdown", "latex", "csv"), default="json")
    args = parser.parse_args()
    rows = count_rows(json.loads(args.input.read_text()), args.confidence)
    if args.format == "json":
        output = json.dumps(rows, indent=2) + "\n"
    elif args.format in ("markdown", "latex"):
        output = count_table(rows, latex=args.format == "latex")
    else:
        import io, csv
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
        output = buffer.getvalue()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)

if __name__ == "__main__":
    main()
