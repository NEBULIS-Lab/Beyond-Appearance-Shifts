import copy
import unittest
from bas_vla.comparisons import build_comparison_jobs


class ComparisonTests(unittest.TestCase):
    def manifest(self):
        return {'carrier': {'name': 'openpi', 'revision': 'a' * 40, 'checkpoint_sha256': 'b' * 64,
                            'normalization_id': 'libero'},
                'seeds': [7, 11], 'episodes_per_seed': 2,
                'tasks': [{'task_id': 'milk', 'init_state_ids': [0, 1], 'init_states_sha256': 'c' * 64,
                           'instructions': {'clean': 'milk', 'control': 'the milk', 'break': 'juice'}}],
                'methods': [{'name': 'strategy', 'revision': 'd' * 40, 'license': 'MIT',
                             'source_url': 'https://github.com/org/repo',
                             'command': ['python', 'adapter.py', '--spec', '{spec}', '--output', '{output}']}]}

    def test_matched_budgets_and_explicit_old_goal(self):
        jobs = build_comparison_jobs(self.manifest())
        self.assertEqual(len(jobs), 6)
        self.assertEqual({tuple(j['spec']['init_state_ids']) for j in jobs}, {(0, 1)})
        self.assertEqual({j['spec']['metric'] for j in jobs}, {'original_goal_success'})
        self.assertEqual({j['spec']['condition'] for j in jobs}, {'clean', 'control', 'break'})

    def test_ambiguous_job_identifiers_rejected(self):
        manifest = self.manifest()
        manifest['methods'][0]['name'] = 'a__b'
        with self.assertRaises(ValueError):
            build_comparison_jobs(manifest)

    def test_require_version_assets_and_unique_seeds(self):
        for key, value in [('revision', None), ('checkpoint_sha256', 'bad')]:
            manifest = self.manifest()
            manifest['carrier'][key] = value
            with self.assertRaises(ValueError): build_comparison_jobs(manifest)
        manifest = self.manifest()
        manifest['seeds'] = [7, 7]
        with self.assertRaises(ValueError): build_comparison_jobs(manifest)

    def test_unsafe_names_and_missing_output_contract(self):
        manifest = self.manifest()
        manifest['methods'][0]['name'] = '../outside'
        with self.assertRaises(ValueError): build_comparison_jobs(manifest)
        manifest = self.manifest()
        manifest['methods'][0]['command'] = ['python', 'adapter.py']
        with self.assertRaises(ValueError): build_comparison_jobs(manifest)


if __name__ == '__main__': unittest.main()
