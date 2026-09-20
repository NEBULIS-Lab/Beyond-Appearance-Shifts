#!/usr/bin/env python3
"""Execute fixed Plus/PRO or non-object manifests with matched trial identities."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bas_vla.evaluation.manifests import asset_path, validate_manifest, verify_assets
from bas_vla.evaluation.dual_goal import rollout_first_hit, selected_indices, stable_seed
from bas_vla.evaluation.nonobject import GoalTracker, SemanticGoalEvaluator
from bas_vla.evaluation.openpi import OpenPIBackend, add_openpi_arguments, load_initial_states


def factory(spec):
    module, name = spec.split(':', 1)
    return getattr(importlib.import_module(module), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_openpi_arguments(parser)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--paper-budget', action='store_true')
    parser.add_argument('--trust-init-files', action='store_true')
    parser.add_argument('--checkpoint-id', help='Loaded carrier artifact identifier; must match manifest')
    parser.add_argument('--normalization-key', help='Loaded checkpoint normalization key; must match manifest')
    parser.add_argument('--backend-factory', help='module:factory(args, manifest) for another matched carrier')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validation = validate_manifest(manifest, paper_budget=args.paper_budget)
    if args.validate_only:
        print(json.dumps(validation, indent=2)); return
    if args.asset_root is None or args.output_dir is None:
        parser.error('--asset-root and --output-dir are required for execution')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('output directory must be empty')
    for key in ('checkpoint_id', 'normalization_key'):
        if getattr(args, key) != manifest['carrier'][key]:
            raise ValueError(f'--{key.replace(chr(95), chr(45))} must match the carrier manifest')
    verify_assets(manifest, args.asset_root)
    if args.backend_factory:
        backend = factory(args.backend_factory)(args, manifest)
        # Alternate carriers must validate their loaded assets/conventions too.
        backend.validate_carrier(manifest['carrier'])
    else:
        if manifest['carrier']['name'] != 'openpi' or manifest['carrier']['config_name'] != args.config_name:
            raise ValueError('built-in backend requires carrier=openpi and a matching config_name')
        backend = OpenPIBackend(args, expected_carrier=manifest['carrier'])
    make_env = factory(manifest['environment_factory'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    terminal = args.output_dir / 'terminal_states'; terminal.mkdir()
    receipts = []
    with (args.output_dir / 'episodes.jsonl').open('w') as stream:
        for case in manifest['cases']:
            states = load_initial_states(asset_path(args.asset_root, case['init_states_file']), allow_pickle=args.trust_init_files)
            for trial in case['trials']:
                seed, idx = trial['seed'], trial['episode_idx']
                selected_indices(len(states), 1, [idx])
                env = make_env(asset_path(args.asset_root, case['bddl_file']), args.resolution,
                    stable_seed(seed, case['case_id'], idx), case.get('env_kwargs', {}))
                try:
                    if manifest['protocol'] == 'nonobject':
                        evaluator = SemanticGoalEvaluator(case['old_goal'], case['new_goal'], case.get('other_goals'))
                    else:
                        tracker = GoalTracker(case['goal'])
                        def evaluator(e):
                            success = tracker.update(e.env._eval_predicate)
                            return {'outcome': 'new' if success else 'none', 'new_hit': success, 'old_hit': False, 'other_hits': []}
                    def plan(obs, step):
                        return backend.predict(obs, case['instruction'], stable_seed(seed, case['case_id'], idx, step), step)
                    row = rollout_first_hit(env, states[idx], plan, evaluator,
                        max_steps=args.max_steps, wait_steps=args.num_steps_wait, replan_steps=args.replan_steps)
                    name = f'{len(receipts):06d}.npz'
                    np.savez_compressed(terminal / name, simulator_state=np.asarray(env.get_sim_state()))
                    row.update(case_id=case['case_id'], category=case['category'], method=args.mode, seed=seed, episode_idx=idx,
                        terminal_state_path=f'terminal_states/{name}', normalization_key=case['normalization_key'])
                    stream.write(json.dumps(row) + '\n'); stream.flush()
                    receipts.append({k:v for k,v in row.items() if k not in {'action_trace','inference_trace'}})
                finally:
                    env.close()
    counts = {outcome: sum(r['outcome'] == outcome for r in receipts) for outcome in ('new','old','other','conflict','timeout')}
    summary = dict(validation, method=args.mode, manifest=manifest, counts=counts,
        success_rate=counts['new']/len(receipts), episodes=receipts,
        arguments={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(args.output_dir / 'summary.json')


if __name__ == '__main__':
    main()
