"""Aggregate explicitly annotated robot trials without inferring unknown labels."""
from collections import defaultdict


def _binary_summary(values):
    observed = [x for x in values if x is not None]
    successes = sum(observed)
    return {'successes': successes, 'n': len(observed),
            'rate': successes / len(observed) if observed else None}


def summarize_trials(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    fields = ('new_success', 'original_success', 'old_target_executed', 'new_target_first_commit')
    for row in records:
        for field in ('trial_id', 'case_id', 'method', 'condition', 'reset_id', 'instruction', 'annotator'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'{field} must be a nonempty string')
        if row['trial_id'] in seen:
            raise ValueError(f'duplicate trial_id: {row["trial_id"]}')
        seen.add(row['trial_id'])
        if row['case_id'] not in ('T1', 'T2', 'T3', 'T4'):
            raise ValueError('case_id must be T1, T2, T3 or T4')
        if row['condition'] not in ('clean', 'control', 'break'):
            raise ValueError('condition must be clean, control or break')
        for field in fields:
            if row.get(field) is not None and type(row[field]) is not bool:
                raise ValueError(f'{field} must be boolean or null')
        if type(row.get('original_subset', False)) is not bool:
            raise ValueError('original_subset must be boolean')
        groups[(row['case_id'], row['method'], row['condition'])].append(row)
    output = []
    for (case, method, condition), rows in sorted(groups.items()):
        suppression = _binary_summary([None if r.get('old_target_executed') is None
                                       else not r['old_target_executed'] for r in rows])
        first_commit = _binary_summary([r.get('new_target_first_commit') for r in rows])
        # Separation uses the same trials for both labels, rather than silently
        # combining rates with different denominators.
        paired = [r for r in rows if r.get('old_target_executed') is not None
                  and r.get('new_target_first_commit') is not None]
        separation = (sum((not r['old_target_executed']) + r['new_target_first_commit']
                          for r in paired) / (2 * len(paired))) if paired else None
        output.append({'case_id': case, 'method': method, 'condition': condition,
                       'trials': len(rows), 'original_subset_trials': sum(r.get('original_subset', False) for r in rows),
                       'new_success': _binary_summary([r.get('new_success') for r in rows]),
                       'original_success': _binary_summary([r.get('original_success') for r in rows]),
                       'old_target_suppression': suppression, 'new_target_first_commit': first_commit,
                       'separation_score': separation, 'separation_n': len(paired)})
    return output
