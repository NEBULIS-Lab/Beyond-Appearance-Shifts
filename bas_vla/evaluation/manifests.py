"""Validation and expansion of portable external/non-object case manifests."""
from __future__ import annotations
from collections import Counter
import hashlib
from pathlib import Path
from .nonobject import validate_goal


def asset_path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('asset paths must be nonempty paths relative to --asset-root')
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('asset path escapes --asset-root')
    return path


def _required_text(obj, key):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip() or value.startswith('REQUIRED'):
        raise ValueError(f'{key} is a required non-placeholder string')


def validate_manifest(manifest, *, paper_budget=False):
    if manifest.get('schema_version') != 'bas_vla_evaluation_v1':
        raise ValueError('unsupported manifest schema')
    if manifest.get('status') != 'ready':
        raise ValueError('manifest is a required-input template; populate it and set status=ready')
    protocol = manifest.get('protocol')
    if protocol not in {'libero_plus', 'libero_pro', 'nonobject'}:
        raise ValueError('protocol must be libero_plus, libero_pro, or nonobject')
    for key in ('benchmark_revision', 'selection_id', 'environment_factory'):
        _required_text(manifest, key)
    carrier = manifest.get('carrier', {})
    for key in ('name', 'checkpoint_id', 'config_name', 'normalization_key'):
        _required_text(carrier, key)
    files = carrier.get('checkpoint_files')
    if not isinstance(files, dict) or not files:
        raise ValueError('carrier.checkpoint_files requires exact weight/normalization file digests')
    for name, digest in files.items():
        asset_path('/checkpoint', name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('checkpoint_files values must be SHA-256 digests')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('manifest needs exact nonempty case list')
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('duplicate case IDs')
    categories = Counter()
    for case in cases:
        for key in ('case_id', 'category', 'instruction', 'bddl_file', 'init_states_file', 'normalization_key'):
            _required_text(case, key)
        if case['normalization_key'] != carrier['normalization_key']:
            raise ValueError('one manifest run must use one explicitly matched normalization key')
        for field in ('bddl_file', 'init_states_file'):
            asset_path('/assets', case[field])
            digest = case.get(field + '_sha256', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'{field}_sha256 must be a SHA-256 digest')
        trials = case.get('trials')
        if not isinstance(trials, list) or not trials:
            raise ValueError('each case requires explicit seed/initial-state trial list')
        keys = []
        for trial in trials:
            if any(type(trial.get(k)) is not int or trial[k] < 0 for k in ('seed', 'episode_idx')):
                raise ValueError('trial seed and episode_idx must be nonnegative integers')
            keys.append((trial['seed'], trial['episode_idx']))
        if len(set(keys)) != len(keys): raise ValueError('duplicate trials within case')
        if protocol == 'nonobject':
            if case['category'] not in {'destination', 'spatial_relation', 'order'}:
                raise ValueError('unknown nonobject category')
            validate_goal(case['old_goal']); validate_goal(case['new_goal'])
            if case['category'] == 'order' and any(case[k]['kind'] != 'ordered' for k in ('old_goal', 'new_goal')):
                raise ValueError('order cases require old and new ordered predicates')
            for goal in case.get('other_goals', {}).values(): validate_goal(goal)
        else:
            validate_goal(case['goal'])
        categories[case['category']] += 1
        if paper_budget:
            expected = 200 if protocol == 'nonobject' else 120
            if len(trials) != expected: raise ValueError(f'paper budget needs {expected} trials per case')
            if protocol == 'nonobject' and sorted(Counter(seed for seed, _ in keys).values()) != [50]*4:
                raise ValueError('nonobject paper protocol requires four seeds by fifty episodes')
    if paper_budget:
        expected_categories = {'light':10, 'background':10, 'noise':10, 'language':10}
        if protocol == 'libero_plus' and dict(categories) != expected_categories:
            raise ValueError('Plus paper protocol requires ten cases in each of light/background/noise/language')
        if protocol == 'libero_pro' and len(cases) != 10:
            raise ValueError('PRO paper protocol requires ten cases')
        if protocol == 'nonobject' and dict(categories) != {'destination':2, 'spatial_relation':2, 'order':2}:
            raise ValueError('nonobject paper protocol requires two tasks per family')
    return {'protocol': protocol, 'cases': len(cases), 'episodes_per_method': sum(len(c['trials']) for c in cases),
            'categories': dict(categories), 'budget_checked': paper_budget}


def verify_assets(manifest, root):
    """Called only by an evaluation run; validation alone never loads assets."""
    for case in manifest['cases']:
        for field in ('bddl_file', 'init_states_file'):
            path = asset_path(root, case[field])
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != case[field + '_sha256']:
                raise ValueError(f'asset digest mismatch: {case["case_id"]}/{field}')


def verify_checkpoint(carrier, root, *, normalization_key):
    """Verify all OpenPI weight files and the actual normalization asset."""
    root = Path(root)
    norm_file = f'assets/{normalization_key}/norm_stats.json'
    if (root / 'model.safetensors').is_file():
        weights = {'model.safetensors'}
    else:
        weights = {str(p.relative_to(root)) for p in (root / 'params').rglob('*') if p.is_file()}
    if not weights:
        raise ValueError('no supported OpenPI checkpoint weights found')
    files = carrier['checkpoint_files']
    if not weights | {norm_file} <= set(files):
        raise ValueError('checkpoint_files must cover every loaded weight and normalization asset')
    for name, expected in files.items():
        digest = hashlib.sha256()
        with asset_path(root, name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'checkpoint digest mismatch: {name}')
