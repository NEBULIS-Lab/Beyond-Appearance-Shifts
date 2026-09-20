"""Matched-carrier experiment specifications for independently installed methods."""
import re


def _identifier(value, name):
    if not isinstance(value, str) or '__' in value or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'{name} must be a portable identifier without a double underscore')
    return value


def _digest(value, length, name):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{' + str(length) + '}', value):
        raise ValueError(f'{name} must be a {length}-character hexadecimal identifier')


def build_comparison_jobs(manifest: dict) -> list[dict]:
    carrier = manifest['carrier']
    _identifier(carrier['name'], 'carrier name')
    _digest(carrier.get('revision'), 40, 'carrier revision')
    _digest(carrier.get('checkpoint_sha256'), 64, 'checkpoint_sha256')
    if not isinstance(carrier.get('normalization_id'), str) or not carrier['normalization_id'].strip():
        raise ValueError('carrier normalization_id is required')
    seeds = manifest['seeds']
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct nonnegative integers')
    count = manifest['episodes_per_seed']
    if type(count) is not int or count <= 0:
        raise ValueError('episodes_per_seed must be a positive integer')
    if not manifest['tasks'] or not manifest['methods']:
        raise ValueError('at least one task and method are required')
    task_ids, method_ids = set(), set()
    for task in manifest['tasks']:
        task_id = _identifier(task['task_id'], 'task_id')
        if task_id in task_ids:
            raise ValueError('duplicate task_id')
        task_ids.add(task_id)
        _digest(task.get('init_states_sha256'), 64, 'init_states_sha256')
        ids = task['init_state_ids']
        if len(ids) != count or any(type(x) is not int or x < 0 for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('init_state_ids must contain exactly episodes_per_seed distinct integer IDs')
        if set(task['instructions']) != {'clean', 'control', 'break'}:
            raise ValueError('each task requires clean/control/break instructions')
        if any(not isinstance(s, str) or not s.strip() for s in task['instructions'].values()):
            raise ValueError('instructions must be nonempty strings')
    jobs = []
    for method in manifest['methods']:
        name = _identifier(method['name'], 'method name')
        if name in method_ids:
            raise ValueError('duplicate method name')
        method_ids.add(name)
        _digest(method.get('revision'), 40, 'method revision')
        if not isinstance(method.get('license'), str) or not method['license'].strip():
            raise ValueError('method license is required')
        if not isinstance(method.get('source_url'), str) or not method['source_url'].startswith('https://'):
            raise ValueError('method source_url must be an HTTPS source URL')
        command = method['command']
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError('method command must be a nonempty argv list')
        if '{spec}' not in command or '{output}' not in command:
            raise ValueError('command must include separate {spec} and {output} arguments')
        for task in manifest['tasks']:
            for seed in seeds:
                for condition, instruction in task['instructions'].items():
                    job_id = f'{name}__{task["task_id"]}__{seed}__{condition}'
                    spec = {'carrier': carrier, 'method': {k: v for k, v in method.items() if k != 'command'},
                            'task_id': task['task_id'], 'seed': seed, 'condition': condition,
                            'instruction': instruction, 'episodes': count,
                            'init_state_ids': task['init_state_ids'],
                            'init_states_sha256': task['init_states_sha256'],
                            'metric': 'original_goal_success'}
                    jobs.append({'job_id': job_id, 'command': command, 'spec': spec})
    return jobs
