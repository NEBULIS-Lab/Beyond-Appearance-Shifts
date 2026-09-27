import hashlib
import tempfile
from pathlib import Path
import unittest
import numpy as np
from bas_vla.evaluation.dual_goal import (aggregate_outcomes, assert_matched_initial_states, classify_hits,
    random_direction_control, rollout_first_hit, selected_indices, state_digest, stable_seed)
from bas_vla.evaluation.nonobject import GoalTracker, SemanticGoalEvaluator
from bas_vla.evaluation.manifests import validate_manifest, asset_path, verify_checkpoint


class FakeEnv:
    def __init__(self, goals):
        self.goals = goals
        self.env = self
    def reset(self): self.index = 0
    def set_init_state(self, state): return {'state': state}
    def get_sim_state(self): return [self.index, 0.0]
    def step(self, action):
        self.index += 1
        return {}, 0, True, {}
    def _eval_predicate(self, predicate): return predicate[0] in self.goals[self.index]


class TestDualGoal(unittest.TestCase):
    def test_conflicts_never_become_new_success(self):
        self.assertEqual(classify_hits(True, True)['outcome'], 'conflict')
        self.assertEqual(classify_hits(False, True, ['third'])['outcome'], 'conflict')
        self.assertEqual(classify_hits(False, False, ['third'])['outcome'], 'other')

    def test_native_done_ignored_until_independent_goal(self):
        env = FakeEnv([set(), set(), {'new'}, {'old'}])
        row = rollout_first_hit(env, [0], lambda obs, step: (np.zeros((1, 7)), {}),
            lambda e: classify_hits(e._eval_predicate(['old']), e._eval_predicate(['new'])),
            max_steps=3, wait_steps=0, replan_steps=1)
        self.assertEqual(row['outcome'], 'new')
        self.assertEqual(row['first_hit_step'], 2)
        self.assertEqual(len(row['inference_trace']), 2)

    def test_wait_steps_also_checked(self):
        env = FakeEnv([set(), {'old'}])
        row = rollout_first_hit(env, [0], lambda *_: self.fail('policy queried during settling'),
            lambda e: classify_hits(e._eval_predicate(['old']), False), max_steps=2, wait_steps=2)
        self.assertEqual(row['outcome'], 'old')
        self.assertEqual(row['action_trace'], [])

    def test_pre_satisfied_goals_rejected(self):
        with self.assertRaises(ValueError):
            rollout_first_hit(FakeEnv([set()]), [0], None, lambda e: classify_hits(True, False), max_steps=1)

    def test_timeout(self):
        row = rollout_first_hit(FakeEnv([set()]*4), [0], lambda *_: (np.zeros((1,7)),{}),
            lambda e: classify_hits(False,False), max_steps=3, wait_steps=0, replan_steps=1)
        self.assertEqual(row['outcome'], 'timeout')
        self.assertIsNone(row['first_hit_step'])

    def test_norm_match_and_gripper(self):
        base = np.zeros((4,7), dtype=np.float32); base[:,6] = -.7
        ref = base.copy(); ref[:,0] = .2; ref[:,6] = 1
        for norm in ('carrier_delta','bas_residual'):
            out, meta = random_direction_control(base, ref, seed=42, norm_reference=norm)
            np.testing.assert_allclose(np.linalg.norm((out-base)[:,:6],axis=1),.2,rtol=1e-6)
            np.testing.assert_array_equal(out[:,6],base[:,6])
            again,_=random_direction_control(base,ref,seed=42,norm_reference=norm)
            np.testing.assert_array_equal(out,again)
            self.assertEqual(meta['norm_reference'],norm)

    def test_clipping_norms_reported_separately(self):
        base=np.ones((1,7)); ref=base+100
        out,meta=random_direction_control(base,ref,seed=1,norm_reference='bas_residual')
        self.assertAlmostEqual(meta['reference_norms'][0],meta['random_preclip_norms'][0],places=4)
        self.assertLess(meta['random_postclip_norms'][0],meta['reference_norms'][0])
        self.assertLessEqual(abs(out[:,:6]).max(),1)
        with self.assertRaises(ValueError): random_direction_control(base,ref[:,:6],seed=1,norm_reference='bas_residual')

    def test_initial_indices_never_silently_truncated(self):
        for idx in ([0,0],[-1],[2]):
            with self.assertRaises(ValueError): selected_indices(2,1,idx)
        with self.assertRaises(ValueError): selected_indices(2,3)
        self.assertEqual(selected_indices(3,2),[0,1])

    def test_matched_hashes_and_aggregate(self):
        common=dict(scene_protocol='native',pair_id='pair',seed=7,episode_idx=0,initial_state_sha256=state_digest([1,2]))
        rows=[dict(common,method='a',outcome='conflict'),dict(common,method='b',outcome='new')]
        assert_matched_initial_states(rows)
        self.assertEqual(aggregate_outcomes(rows)[0]['three_way_counts']['other_timeout'],1)
        with self.assertRaises(ValueError): aggregate_outcomes(rows+rows)
        rows[1]['initial_state_sha256']='different'
        with self.assertRaises(ValueError): assert_matched_initial_states(rows)
        self.assertNotEqual(stable_seed(1,0,101),stable_seed(1,1,1))


class TestOrder(unittest.TestCase):
    def tracker(self): return GoalTracker({'kind':'ordered','predicates':[['a','x'],['b','x']]})
    def test_correct_order_with_persistent_first_predicate(self):
        tracker=self.tracker()
        self.assertFalse(tracker.update(lambda p:False))
        self.assertFalse(tracker.update(lambda p:p[0]=='a'))
        self.assertTrue(tracker.update(lambda p:True))
        self.assertEqual([e['subgoal'] for e in tracker.events],[0,1])
    def test_reverse_and_simultaneous_order_fail(self):
        for first in ({'b'},{'a','b'}):
            tracker=self.tracker()
            self.assertFalse(tracker.update(lambda p:p[0] in first))
            self.assertFalse(tracker.update(lambda p:True))
            self.assertTrue(tracker.violated)
    def test_initial_partial_order_rejected(self):
        spec={'kind':'ordered','predicates':[['a','x'],['b','x']]}
        reverse={'kind':'ordered','predicates':[['b','x'],['a','x']]}
        with self.assertRaises(ValueError):
            rollout_first_hit(FakeEnv([{'a'}]),[0],None,SemanticGoalEvaluator(spec,reverse),max_steps=1)


class TestManifest(unittest.TestCase):
    def manifest(self):
        return {'schema_version':'bas_vla_evaluation_v1','status':'ready','protocol':'libero_pro',
            'benchmark_revision':'abc123','selection_id':'small-test','environment_factory':'module:factory',
            'carrier':{'name':'openpi','checkpoint_id':'model-id','config_name':'pi05_libero','normalization_key':'libero','checkpoint_files':{'model.safetensors':'f'*64}},
            'cases':[{'case_id':'one','category':'task','instruction':'put object in basket','bddl_file':'task.bddl',
                'init_states_file':'task.npy','bddl_file_sha256':'0'*64,'init_states_file_sha256':'1'*64,
                'normalization_key':'libero','trials':[{'seed':7,'episode_idx':0}],
                'goal':{'kind':'all','predicates':[['in','object','basket']]}}]}
    def test_valid_custom_manifest(self):
        self.assertEqual(validate_manifest(self.manifest())['episodes_per_method'],1)
    def test_templates_and_wrong_paper_budget_rejected(self):
        m=self.manifest();m['status']='requires_artifacts'
        with self.assertRaises(ValueError): validate_manifest(m)
        with self.assertRaises(ValueError): validate_manifest(self.manifest(),paper_budget=True)
    def test_duplicates_paths_and_normalization_rejected(self):
        m=self.manifest();m['cases'][0]['trials']*=2
        with self.assertRaises(ValueError): validate_manifest(m)
        m=self.manifest();m['cases'][0]['normalization_key']='other'
        with self.assertRaises(ValueError): validate_manifest(m)
        for value in ('../outside','/absolute'):
            with self.assertRaises(ValueError): asset_path('/assets',value)
    def test_checkpoint_weight_and_normalization_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            weights=root/'model.safetensors'; weights.write_bytes(b'fake-weights')
            norm=root/'assets/libero/norm_stats.json'; norm.parent.mkdir(parents=True); norm.write_text('{}')
            files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (weights,norm)}
            verify_checkpoint({'checkpoint_files':files}, root, normalization_key='libero')
            with self.assertRaises(ValueError): verify_checkpoint({'checkpoint_files':files},root,normalization_key='wrong')
            weights.write_bytes(b'other-weights')
            with self.assertRaises(ValueError): verify_checkpoint({'checkpoint_files':files},root,normalization_key='libero')

    def test_order_family_must_use_ordered_predicates(self):
        m=self.manifest();m['protocol']='nonobject';c=m['cases'][0];c['category']='order'
        c['old_goal']=c['new_goal']=c['goal']
        with self.assertRaises(ValueError): validate_manifest(m)


if __name__=='__main__': unittest.main()
