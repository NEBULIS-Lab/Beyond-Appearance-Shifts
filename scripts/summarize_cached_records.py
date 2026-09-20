#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.records import load_cached_records, summarize_cached_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize cached BAS-VLA breaking-training records before public release."
    )
    parser.add_argument("--records-path", type=Path, required=True, help="Path to a JSON list of cached records.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_cached_records(load_cached_records(args.records_path))
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
