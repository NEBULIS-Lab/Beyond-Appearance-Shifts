#!/usr/bin/env python3
"""Create matched evaluation specs; optionally execute user-supplied method adapters."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bas_vla.comparisons import build_comparison_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Execute adapters after writing specs.')
    args = parser.parse_args()
    jobs = build_comparison_jobs(json.loads(args.manifest.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for job in jobs:
        folder = args.output_dir / job['job_id']
        folder.mkdir(exist_ok=False)
        spec, output = (folder / 'spec.json').resolve(), (folder / 'result.json').resolve()
        spec.write_text(json.dumps(job['spec'], indent=2) + '\n')
        replacements = {'{spec}': str(spec), '{output}': str(output)}
        argv = [replacements.get(item, item) for item in job['command']]
        plan.append({'job_id': job['job_id'], 'argv': argv})
    (args.output_dir / 'commands.json').write_text(json.dumps(plan, indent=2) + '\n')
    if args.execute:
        for item in plan:
            subprocess.run(item['argv'], check=True)
    print(f'Prepared {len(plan)} matched condition runs in {args.output_dir}')


if __name__ == '__main__':
    main()
