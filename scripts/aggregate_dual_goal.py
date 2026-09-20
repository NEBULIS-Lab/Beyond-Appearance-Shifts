#!/usr/bin/env python3
"""Aggregate dual-goal JSONL files while checking matched episode identities."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.evaluation.dual_goal import aggregate_outcomes, assert_matched_initial_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-methods', nargs='+', help='Require complete method pairing for every episode')
    args = parser.parse_args()
    records = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
    if not records: raise ValueError('no episode records')
    assert_matched_initial_states(records)
    if args.require_methods:
        groups = {}
        for row in records:
            key = tuple(row[k] for k in ('scene_protocol','pair_id','seed','episode_idx'))
            groups.setdefault(key,set()).add(row['method'])
        if any(methods != set(args.require_methods) for methods in groups.values()):
            raise ValueError('incomplete or unexpected method coverage in matched episodes')
    summary = {'rows':aggregate_outcomes(records), 'num_records':len(records)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
