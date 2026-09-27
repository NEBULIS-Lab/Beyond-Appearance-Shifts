import unittest
import numpy as np
from bas_vla.analysis.wilson import wilson, count_rows, count_table
from bas_vla.analysis.gate import summarize_gate, probe_better
from bas_vla.analysis.latency import summarize_latency
from bas_vla.analysis.masks import corrupt_mask, mask_iou, summarize_mask_outcomes


class WilsonTests(unittest.TestCase):
    def test_known_bounds(self):
        self.assertAlmostEqual(wilson(0, 200)[1], 0.018845326377266575)
        self.assertAlmostEqual(wilson(200, 200)[0], 1-wilson(0, 200)[1])
        self.assertAlmostEqual(wilson(50, 100)[0], 0.4038315303659956)
    def test_invalid_counts(self):
        for args in ((0,0),(-1,10),(11,10),(1.5,10),(True,10)):
            with self.assertRaises(ValueError): wilson(*args)
    def test_table_escape(self):
        rows = count_rows([{"label":"a_b & c", "successes":1, "total":2}])
        self.assertIn(r"a\_b \& c", count_table(rows, latex=True))
        self.assertIn("50.0", count_table(rows))


class GateTests(unittest.TestCase):
    def row(self, i, gate=1., probe=0.):
        return dict(task_id="t", episode_id=i, chunk_id=0, semantic_changed=False,
                    visual_shift=False, gate=gate, base_action=[1.], probe_action=[probe], reference_action=[0.])
    def test_target_ties_and_confusion_denominators(self):
        self.assertFalse(probe_better([1],[-1],[0]))
        rows=[self.row(0,1.,2.), self.row(1,0.,0.), self.row(2,1.,0.)]
        group=summarize_gate(rows)["matrix"][0]
        self.assertEqual(group["false_activation_count"],1)
        self.assertEqual(group["false_negative_count"],1)
        self.assertEqual(group["false_positive_rate"],1.)
        self.assertEqual(group["false_negative_rate"],0.5)
        self.assertAlmostEqual(group["false_activation_fraction_all"],1/3)
        self.assertIsNone(summarize_gate(rows)["matrix"][1]["false_positive_rate"])
    def test_oracle_separate_and_no_chunk_retention(self):
        row=self.row(0);row.update(oracle_router_success=True,full_success=False)
        with self.assertRaises(ValueError): summarize_gate([row])
        result=summarize_gate([row],unit="episode")["matrix"][0]
        self.assertEqual(result["oracle_router_success_rate"],1.)
        self.assertEqual(result["full_success_rate"],0.)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): summarize_gate([self.row(0),self.row(0)])


class MaskTests(unittest.TestCase):
    def test_iou_and_random_area(self):
        mask=np.zeros((10,10),bool);mask[:4]=True
        corrupted,meta=corrupt_mask(mask,kind="iou",target_iou=0.5,seed=7)
        self.assertEqual(mask_iou(mask,corrupted),0.5)
        self.assertEqual(meta["achieved_iou"],0.5)
        random,_=corrupt_mask(mask,kind="random",seed=7)
        self.assertEqual(random.sum(),mask.sum())
        np.testing.assert_array_equal(random,corrupt_mask(mask,kind="random",seed=7)[0])
    def test_missed_target(self):
        mask=np.ones((4,4),bool);target=np.zeros_like(mask);target[0]=True
        result,_=corrupt_mask(mask,kind="missed_target",target_mask=target,seed=1)
        self.assertFalse(result[target].any())
        self.assertTrue(result[~target].all())
        with self.assertRaises(ValueError):corrupt_mask(mask,kind="missed_target",seed=1)
    def test_paired_counts_and_reference_consistency(self):
        rows=[dict(task_id="t",corruption="iou0.5",seed=7,episode_id=0,success=False,normal_success=True)]
        result=summarize_mask_outcomes(rows)[0]
        self.assertEqual(result["delta_percentage_points"],-100.)
        self.assertFalse(result["protocol_complete"])
        with self.assertRaises(ValueError):summarize_mask_outcomes(rows + [{**rows[0],"corruption":"random","normal_success":False}])


class LatencyTests(unittest.TestCase):
    def test_runtime_trace_export_keeps_sync_and_requires_pass(self):
        from bas_vla.records.diagnostics import timing_rows_from_episodes
        trace=dict(mode="full",chunk_id=0,warmup=False,synchronized=False,timing_ms={"end_to_end":1.},task_id=None,pass_id=None)
        episodes=[dict(pair_id="t",seed=7,episode_idx=0,inference_trace=[trace])]
        with self.assertRaises(ValueError):timing_rows_from_episodes(episodes)
        row=timing_rows_from_episodes(episodes,pass_id="pass1")[0]
        self.assertEqual(row["task_id"],"t")
        self.assertFalse(row["synchronized"])
        self.assertEqual(row["episode_id"],"7:0:t:")
    def test_missing_pass_rejected(self):
        row=self.rows()[0];row["pass_id"]=None
        with self.assertRaises(ValueError):summarize_latency([row])
    def rows(self):
        return [dict(task_id="t",mode="full",pass_id=p,chunk_id=i,warmup=False,synchronized=True,
                     timing_ms={"end_to_end":float(i+1),"base":float(i+2),"nested":5.})
                for p in range(3) for i in range(200)]
    def test_complete_quantiles_and_components(self):
        rows=self.rows();rows.append({**rows[0],"warmup":True,"timing_ms":{"end_to_end":9999.}})
        result=summarize_latency(rows)
        group=result["by_task_mode"][0]
        self.assertTrue(group["protocol_complete"])
        self.assertEqual(result["warmup_rows_excluded"],1)
        self.assertEqual(group["metrics"]["end_to_end"]["p50_ms"],100.5)
        self.assertAlmostEqual(group["metrics"]["end_to_end"]["p95_ms"],190.05)
    def test_incomplete_unsynchronized_duplicates(self):
        rows=self.rows()[:2]
        self.assertFalse(summarize_latency(rows)["by_task_mode"][0]["protocol_complete"])
        with self.assertRaises(ValueError):summarize_latency(rows + [rows[0]])
        rows[0]["synchronized"]=False
        with self.assertRaises(ValueError):summarize_latency(rows)
        self.assertEqual(summarize_latency(rows,allow_unsynchronized=True)["by_task_mode"][0]["unsynchronized_chunks"],1)

if __name__ == "__main__":unittest.main()
