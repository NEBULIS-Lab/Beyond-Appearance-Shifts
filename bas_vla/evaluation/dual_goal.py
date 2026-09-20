"""First-hit dual-goal outcomes and explicitly norm-referenced random controls."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
from typing import Any, Callable
import numpy as np


def classify_hits(old_hit: bool, new_hit: bool, other_hits=()) -> dict[str, Any]:
    others = list(other_hits)
    count = int(old_hit) + int(new_hit) + len(others)
    outcome = 'none' if not count else ('conflict' if count > 1 else (
        'new' if new_hit else 'old' if old_hit else 'other'))
    return dict(outcome=outcome, old_hit=bool(old_hit), new_hit=bool(new_hit), other_hits=others)


def scene_objects(env) -> list[str]:
    return sorted(name[:-2] if name.endswith('_1') else name
                  for name in env.env.objects_dict if name != 'basket_1')


def evaluate_goals(env, old_object: str, new_object: str, third_objects: list[str]):
    def hit(name):
        return bool(env.env._eval_predicate(['in', f'{name}_1', 'basket_1_contain_region']))
    return classify_hits(hit(old_object), hit(new_object), [x for x in third_objects if hit(x)])


def selected_indices(state_count: int, count: int, indices=None) -> list[int]:
    result = list(range(count)) if indices is None else list(indices)
    if not result or len(result) != len(set(result)):
        raise ValueError('episode indices must be nonempty and unique')
    if any(type(i) is not int or i < 0 or i >= state_count for i in result):
        raise ValueError(f'episode index outside available initial states (n={state_count})')
    return result


def state_digest(state) -> str:
    array = np.asarray(state, dtype='<f8')
    if not np.isfinite(array).all():
        raise ValueError('nonfinite simulator state')
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def stable_seed(*parts) -> int:
    # Avoid overlap between episode, task and step fields in arithmetic seeds.
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], 'little')


def action_chunk(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 7 or not np.isfinite(array).all():
        raise ValueError('LIBERO actions must be finite nonempty (horizon, 7) chunks')
    return array


def random_direction_control(base, reference, *, seed: int, norm_reference: str):
    """Match each action's six continuous dimensions before output clipping.

    carrier_delta: base=old instruction, reference=changed instruction.
    bas_residual: base=changed instruction, reference=calibrated changed action.
    The caller must supply the *unclipped* calibrated output to match its raw
    residual. This function records both reference and post-clipping norms.
    """
    if norm_reference not in {'carrier_delta', 'bas_residual'}:
        raise ValueError('norm_reference must be carrier_delta or bas_residual')
    base, reference = action_chunk(base), action_chunk(reference)
    if base.shape != reference.shape:
        raise ValueError('norm matching requires identical action chunk shapes')
    delta_norms = np.linalg.norm(reference[:, :6] - base[:, :6], axis=1)
    direction = np.random.default_rng(seed).normal(size=(len(base), 6))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    delta = direction * delta_norms[:, None]
    result = base.copy()
    result[:, :6] = np.clip(base[:, :6] + delta, -1, 1)
    return result, {'norm_reference': norm_reference, 'continuous_dimensions': list(range(6)),
                    'gripper_source': 'old_instruction' if norm_reference == 'carrier_delta' else 'changed_instruction',
                    'reference_norms': delta_norms.tolist(),
                    'random_preclip_norms': np.linalg.norm(delta, axis=1).tolist(),
                    'random_postclip_norms': np.linalg.norm(result[:, :6] - base[:, :6], axis=1).tolist()}


def rollout_first_hit(env, initial_state, planner: Callable, evaluate: Callable, *,
                      max_steps: int, wait_steps: int = 10, replan_steps: int = 5):
    """Check independent goals after *every* step, including settling steps.

    Native ``done`` is diagnostic only: it may encode the old goal. A conflict
    stops the episode and joins Other/timeout in the three-way summary.
    """
    if max_steps < 1 or wait_steps < 0 or replan_steps < 1:
        raise ValueError('invalid rollout budget')
    env.reset()
    obs = env.set_init_state(initial_state)
    initial = evaluate(env)
    if initial['outcome'] != 'none' or initial.get('initial_subgoal_satisfied', False):
        raise ValueError(f'goal already satisfied at initialization: {initial}')
    digest = state_digest(env.get_sim_state())
    queue = deque()
    actions, traces = [], []
    hit = initial
    native_done = False
    for step in range(max_steps + wait_steps):
        if step < wait_steps:
            action = [0., 0., 0., 0., 0., 0., -1.]
        else:
            if not queue:
                chunk, trace = planner(obs, step - wait_steps)
                chunk = action_chunk(chunk)
                if len(chunk) < replan_steps:
                    raise ValueError('policy horizon shorter than replan_steps')
                queue.extend(chunk[:replan_steps])
                traces.append(dict(trace, step=step - wait_steps, actions=chunk.tolist()))
            action = np.asarray(queue.popleft()).tolist()
            actions.append(action)
        obs, _, done, _ = env.step(action)
        native_done |= bool(done)
        hit = evaluate(env)
        if hit['outcome'] != 'none':
            break
    outcome = hit['outcome'] if hit['outcome'] != 'none' else 'timeout'
    return {'outcome': outcome, 'first_hit': hit,
            'first_hit_step': step + 1 if outcome != 'timeout' else None,
            'steps': step + 1, 'native_done_seen': native_done,
            'initial_state_sha256': digest, 'action_trace': actions, 'inference_trace': traces}


def aggregate_outcomes(records):
    groups, keys = defaultdict(Counter), set()
    specifications = {}
    for row in records:
        key = tuple(row[k] for k in ('scene_protocol', 'pair_id', 'method', 'seed', 'episode_idx'))
        if key in keys:
            raise ValueError(f'duplicate episode: {key}')
        keys.add(key)
        specification = tuple(row.get(k) for k in ('mode', 'random_norm_reference', 'image_shift_preset', 'policy_noise_mode'))
        if key[:3] in specifications and specifications[key[:3]] != specification:
            raise ValueError('cannot pool different modes, norm references, visual shifts, or noise protocols')
        specifications[key[:3]] = specification
        outcome = row['outcome']
        if outcome not in {'new', 'old', 'other', 'conflict', 'timeout'}:
            raise ValueError(f'invalid terminal outcome: {outcome}')
        groups[key[:3]][outcome] += 1
    output = []
    for (protocol, pair, method), counts in sorted(groups.items()):
        n = sum(counts.values())
        three = {'new': counts['new'], 'old': counts['old'],
                 'other_timeout': counts['other'] + counts['conflict'] + counts['timeout']}
        output.append(dict(scene_protocol=protocol, pair_id=pair, method=method, n=n,
                           counts=dict(counts), three_way_counts=three,
                           rates={k: v / n for k, v in three.items()}))
    return output


def assert_matched_initial_states(records):
    hashes = {}
    for row in records:
        key = tuple(row[k] for k in ('scene_protocol', 'pair_id', 'seed', 'episode_idx'))
        if key in hashes and hashes[key] != row['initial_state_sha256']:
            raise ValueError(f'mismatched method initial states: {key}')
        hashes[key] = row['initial_state_sha256']
