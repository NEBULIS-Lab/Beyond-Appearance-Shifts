#!/usr/bin/env python3
"""Summarize a JSON list of human-annotated robot trials (no hardware access)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.deployment.trials import summarize_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--original-subset-only', action='store_true')
    args = parser.parse_args()
    rows = json.loads(args.records.read_text())
    if not isinstance(rows, list):
        raise ValueError('records must be a JSON list')
    # Validate before filtering, including duplicates and boolean annotations.
    summary = summarize_trials(rows)
    if args.original_subset_only:
        summary = summarize_trials([r for r in rows if r.get('original_subset', False)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
