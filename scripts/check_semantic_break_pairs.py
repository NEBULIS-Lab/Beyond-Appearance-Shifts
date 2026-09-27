#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.pairs import load_semantic_break_pairs, summarize_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a BAS-VLA semantic break pair configuration."
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        required=True,
        help="Path to a semantic break pair JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = load_semantic_break_pairs(args.pairs_path)
    summary = summarize_pairs(pairs)

    if summary["duplicate_pair_ids"]:
        raise RuntimeError(
            "pair config contains duplicate pair_id values: "
            + ", ".join(summary["duplicate_pair_ids"])
        )

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
