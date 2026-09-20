#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.analysis.bootstrap import bootstrap_mean, bootstrap_paired_difference, bootstrap_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bootstrap summaries from JSON rows.")
    parser.add_argument("--input-path", type=Path, required=True, help="JSON list or object containing a `rows` list.")
    parser.add_argument("--column", required=True, help="Target metric column.")
    parser.add_argument("--reference-column", default=None, help="Optional paired reference column.")
    parser.add_argument("--binary", action="store_true", help="Treat the target column as binary.")
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("input JSON must be a list or an object with a `rows` list")
    return rows


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input_path)
    values = [row[args.column] for row in rows]
    if args.reference_column:
        reference = [row[args.reference_column] for row in rows]
        summary = bootstrap_paired_difference(
            reference,
            values,
            num_bootstrap=args.num_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
    elif args.binary:
        summary = bootstrap_rate(
            values,
            num_bootstrap=args.num_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
    else:
        summary = bootstrap_mean(
            values,
            num_bootstrap=args.num_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )

    payload = {
        "column": args.column,
        "reference_column": args.reference_column,
        "binary": args.binary,
        "summary": {
            "mean": summary.mean,
            "ci_low": summary.ci_low,
            "ci_high": summary.ci_high,
            "num_samples": summary.num_samples,
            "num_bootstrap": summary.num_bootstrap,
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
