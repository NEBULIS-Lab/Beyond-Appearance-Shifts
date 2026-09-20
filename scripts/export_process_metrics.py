#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.analysis.process_metrics import build_process_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export curve-ready process metrics from paired condition summaries.")
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--variant-summary", required=True)
    parser.add_argument("--compare-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    reference_summary = load_json(args.reference_summary)
    variant_summary = load_json(args.variant_summary)
    compare_payload = load_json(args.compare_json)
    rows = build_process_metric_rows(reference_summary, variant_summary, compare_payload.get("rows", []))

    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "reference_summary": str(Path(args.reference_summary).resolve()),
        "variant_summary": str(Path(args.variant_summary).resolve()),
        "compare_json": str(Path(args.compare_json).resolve()),
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
