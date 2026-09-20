#!/usr/bin/env python3
"""Matched first-hit object-swap evaluation with explicit random-control norms."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bas_vla.evaluation.dual_goal import (action_chunk, aggregate_outcomes, assert_matched_initial_states,
    evaluate_goals, random_direction_control, rollout_first_hit, scene_objects, selected_indices, stable_seed)
from bas_vla.evaluation.openpi import OpenPIBackend, add_openpi_arguments, create_env, load_initial_states


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_openpi_arguments(parser)
    parser.add_argument('--pairs-config', type=Path, required=True)
    parser.add_argument('--pair-ids', nargs='+')
    parser.add_argument('--methods', nargs='+', choices=['old_instruction', 'changed_instruction', 'bas', 'random_direction'], default=['old_instruction', 'changed_instruction'])
    parser.add_argument('--random-norm-reference', choices=['carrier_delta', 'bas_residual'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[7])
    parser.add_argument('--num-trials-per-pair', type=int, default=50)
    parser.add_argument('--episode-indices', type=int, nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('methods and seeds must be unique')
    if 'random_direction' in args.methods and args.random_norm_reference is None:
        raise ValueError('random_direction requires --random-norm-reference')
    if 'bas' in args.methods and args.mode not in {'core', 'pres', 'full'}:
        raise ValueError('bas requires --mode core, pres, or full')
    config = json.loads(args.pairs_config.read_text())
    pairs = config['pairs']
    if args.pair_ids:
        missing = set(args.pair_ids) - {p['pair_id'] for p in pairs}
        if missing: raise ValueError(f'unknown pair IDs: {sorted(missing)}')
        pairs = [p for p in pairs if p['pair_id'] in args.pair_ids]
    if not pairs or len({p['pair_id'] for p in pairs}) != len(pairs):
        raise ValueError('pair IDs must be nonempty and unique')
    if any(p['old_object'] == p['new_object'] for p in pairs):
        raise ValueError('old and new objects must differ')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('output directory must be empty')
    backend = OpenPIBackend(args)
    if args.random_norm_reference == 'bas_residual' and backend.runtime.calibrator is None:
        from bas_vla.runtime.calibration import ResidualCalibrator
        backend.runtime.calibrator = ResidualCalibrator.from_args(args, backend.policy, 'libero_env')
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[config['suite']]()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    terminal = args.output_dir / 'terminal_states'
    terminal.mkdir()
    records = []
    with (args.output_dir / 'episodes.jsonl').open('w') as stream:
        for pair in pairs:
            task = suite.get_task(int(pair['task_id']))
            states = load_initial_states(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file, allow_pickle=True)
            indices = selected_indices(len(states), args.num_trials_per_pair, args.episode_indices)
            override = pair.get('bddl_override')
            bddl = (args.pairs_config.parent / override).resolve() if override else Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            if not bddl.is_file(): raise FileNotFoundError(bddl)
            for seed in args.seeds:
                for idx in indices:
                    for method in args.methods:
                        # A fresh, identically seeded environment isolates method order.
                        env = create_env(bddl, args.resolution, stable_seed(seed, pair['pair_id'], idx))
                        try:
                            objects = scene_objects(env)
                            if not {pair['old_object'], pair['new_object']} <= set(objects):
                                raise ValueError(f"pair {pair['pair_id']} needs co-present targets: {objects}")
                            thirds = [o for o in objects if o not in {pair['old_object'], pair['new_object']}]
                            def plan(obs, step):
                                noise = stable_seed(seed, pair['pair_id'], idx, step)
                                if method == 'bas':
                                    actions, trace = backend.predict(obs, pair['new_instruction'], noise, step)
                                    trace.update(policy_noise_mode=args.policy_noise_mode, policy_noise_seed=noise, final_clip=[-1.0, 1.0])
                                    return np.clip(action_chunk(actions), -1, 1), trace
                                view = backend.observation(obs, noise)
                                prompt = pair['old_instruction'] if method in {'old_instruction', 'random_direction'} else pair['new_instruction']
                                if method == 'random_direction' and args.random_norm_reference == 'bas_residual':
                                    prompt = pair['new_instruction']
                                base, timing = backend.infer(view, prompt, noise)
                                meta = dict(method=method, policy_noise_mode=args.policy_noise_mode, policy_noise_seed=noise, timing=timing)
                                if method == 'random_direction':
                                    if args.random_norm_reference == 'carrier_delta':
                                        reference, _ = backend.infer(view, pair['new_instruction'], noise)
                                    else:
                                        reference = backend.runtime.calibrator.apply(base, pair['new_instruction'])
                                    base, norm_meta = random_direction_control(base, reference, seed=stable_seed(noise, 'random'), norm_reference=args.random_norm_reference)
                                    meta.update(norm_meta)
                                return action_chunk(base), meta
                            started = time.perf_counter()
                            row = rollout_first_hit(env, states[idx], plan,
                                lambda e: evaluate_goals(e, pair['old_object'], pair['new_object'], thirds),
                                max_steps=args.max_steps, wait_steps=args.num_steps_wait, replan_steps=args.replan_steps)
                            filename = f'{len(records):06d}.npz'
                            np.savez_compressed(terminal / filename, simulator_state=np.asarray(env.get_sim_state()))
                            row.update(pair_id=pair['pair_id'], scene_protocol=pair['scene_protocol'], task_id=pair['task_id'],
                                method=method, mode=args.mode if method == 'bas' else 'frozen',
                                random_norm_reference=args.random_norm_reference if method == 'random_direction' else None,
                                image_shift_preset=args.image_shift_preset, policy_noise_mode=args.policy_noise_mode,
                                seed=seed, episode_idx=idx, third_objects=thirds,
                                terminal_state_path=f'terminal_states/{filename}', wall_seconds=time.perf_counter()-started)
                            records.append(row)
                            assert_matched_initial_states(records)
                            stream.write(json.dumps(row) + '\n'); stream.flush()
                        finally:
                            env.close()
    summary = {'protocol': config['protocol'], 'mode': args.mode, 'random_norm_reference': args.random_norm_reference,
               'checkpoint': str(args.checkpoint_dir), 'config_name': args.config_name,
               'pairs': pairs, 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               'rows': aggregate_outcomes(records)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(args.output_dir / 'summary.json')


if __name__ == '__main__':
    main()
