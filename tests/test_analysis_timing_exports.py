import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bas_vla.analysis.latency import summarize_latency
from bas_vla.records.diagnostics import timing_rows_from_episodes

ROOT = Path(__file__).resolve().parents[1]


def trace(chunk_id=0):
    return dict(mode="full", chunk_id=chunk_id, warmup=False, synchronized=False,
                timing_ms={"end_to_end":12.0,"base":5.0}, task_id=4, pass_id=None)


class EvaluatorTimingExportTests(unittest.TestCase):
    def test_openpi_summary_chunk_trace_inherits_seed(self):
        payload = {"seed":7,"per_episode":[{"task_id":4,"episode_idx":0,"pair_id":"swap", "instruction_tag":"break", "chunk_trace":[trace()]}]}
        rows = timing_rows_from_episodes(payload,pass_id="p1")
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["source_trace"],"chunk_trace")
        self.assertEqual(rows[0]["episode_id"],"7:0:swap:break")
        self.assertFalse(rows[0]["synchronized"])
        self.assertEqual(rows[0]["timing_ms"],trace()["timing_ms"])

    def test_appearance_summary_preserving_trace(self):
        payload = {"seed":11,"per_episode":[{"task_id":4,"episode_idx":3,"shift_preset":"dark","preserving_trace":[trace(2),trace(3)]}]}
        rows = timing_rows_from_episodes(payload,pass_id="p2")
        self.assertEqual([r["chunk_id"] for r in rows],[2,3])
        self.assertEqual(rows[0]["episode_id"],"11:3")
        self.assertEqual(rows[0]["source_trace"],"preserving_trace")

    def test_oft_triplets_survive_restarted_chunk_indices(self):
        record = dict(task_id=4,episode_idx=0,pair_id="swap")
        record.update({f"{name}_preserving_trace":[trace()] for name in ("clean","control","break")})
        rows = timing_rows_from_episodes({"records":[record]},pass_id="p3")
        self.assertEqual([r["condition"] for r in rows],["clean","control","break"])
        self.assertEqual(len({r["episode_id"] for r in rows}),3)
        result = summarize_latency(rows,allow_unsynchronized=True)
        self.assertEqual(result["by_task_mode"][0]["metrics"]["end_to_end"]["n"],3)
        self.assertFalse(result["by_task_mode"][0]["protocol_complete"])

    def test_unknown_or_legacy_formats_fail_with_contract(self):
        with self.assertRaisesRegex(ValueError,"episodes, per_episode, or records"):
            timing_rows_from_episodes({"unknown":[]},pass_id="p")
        with self.assertRaisesRegex(ValueError,"legacy traces"):
            timing_rows_from_episodes({"per_episode":[{"task_id":4,"episode_idx":0,"chunk_trace":[{"mode":"full"}]}]},pass_id="p")

    def test_cli_accepts_actual_summary_wrapper(self):
        payload = {"seed":19,"per_episode":[{"task_id":4,"episode_idx":1,"preserving_trace":[trace()]}]}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"summary.json";output=Path(directory)/"timings.json"
            source.write_text(json.dumps(payload))
            result=subprocess.run([sys.executable,str(ROOT/"scripts/export_timing_records.py"),"--input",str(source),"--output",str(output),"--pass-id","p1"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(output.read_text())[0]["episode_id"],"19:1")

if __name__ == "__main__": unittest.main()
