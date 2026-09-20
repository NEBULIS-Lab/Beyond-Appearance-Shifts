import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from bas_vla.breaking.training import training_config, compute_losses, build_vocab_from_records, attach_bow_features, build_tensor_dataset
from bas_vla.records.cached_records import validate_cached_records, export_triplet_records

ROOT=Path(__file__).resolve().parents[1]

class Identity:
    def __call__(self,action,features):return action


class TrainingObjectiveTests(unittest.TestCase):
    def batch(self):
        return dict(clean_base=torch.tensor([[1.,2.]]),control_base=torch.tensor([[2.,3.]]),
                    break_base=torch.tensor([[1.,2.]]),expert=torch.zeros((1,2)),
                    clean_feat=torch.zeros((1,1)),control_feat=torch.zeros((1,1)),break_feat=torch.zeros((1,1)))
    def test_paper_formula(self):
        cfg=training_config("paper_three_component")
        loss,metrics,_=compute_losses(Identity(),self.batch(),cfg)
        # imit=5+13; cons=2; sep=margin+sqrt(2)
        self.assertAlmostEqual(loss.item(),18+0.5*2+0.15+np.sqrt(2),places=5)
        self.assertEqual(cfg.lambda_anchor,0)
        self.assertEqual(cfg.lambda_expert_margin,0)
    def test_legacy_formula_unchanged(self):
        cfg=training_config("legacy")
        loss,m,_=compute_losses(Identity(),self.batch(),cfg)
        expected=m["loss_clean_mse"]+m["loss_control_mse"]+0.5*m["loss_consistency"]+m["loss_margin_clean"]+m["loss_margin_expert"]+0.2*m["loss_anchor"]
        self.assertAlmostEqual(loss.item(),expected,places=5)
    def test_ablation_and_config_validation(self):
        self.assertEqual(training_config("paper_without_consistency").lambda_consistency,0)
        self.assertEqual(training_config("paper_without_separation").lambda_margin,0)
        with self.assertRaises(ValueError):training_config("unknown")
        with self.assertRaises(ValueError):training_config(epochs=0)
        with self.assertRaises(ValueError):training_config(lambda_anchor=-1)


class RecordTests(unittest.TestCase):
    def records(self):return json.loads((ROOT/"examples/records/synthetic_triplets.json").read_text())
    def test_example_and_export(self):
        records=self.records()
        self.assertEqual(validate_cached_records(records),7)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"triplets.jsonl";dest=Path(directory)/"records.json"
            source.write_text("\n".join(json.dumps(r) for r in records))
            self.assertEqual(export_triplet_records(source,dest,jsonl=True)["num_records"],2)
            self.assertEqual(json.loads(dest.read_text()),records)
        features=attach_bow_features(records,build_vocab_from_records(records[:1]))
        self.assertEqual(tuple(build_tensor_dataset(features)["expert"].shape),(2,7))
    def test_nonfinite_shape_and_splits_rejected(self):
        for problem in ("nan","shape","split"):
            rows=self.records()
            if problem=="nan":rows[0]["expert_action"][0]=float("nan")
            if problem=="shape":rows[0]["expert_action"].append(0.)
            if problem=="split":rows[1]["split"]="train"
            with self.assertRaises(ValueError):validate_cached_records(rows)
    def test_source_state_cannot_cross_splits(self):
        rows=self.records();rows[1]["source_state_id"]=rows[0]["source_state_id"]
        with self.assertRaises(ValueError):validate_cached_records(rows)
    def test_shipped_configs_resolve(self):
        for path in (ROOT/"configs/training").glob("*.json"):
            data=json.loads(path.read_text())
            cfg=training_config(data["objective_preset"],**data["overrides"])
            self.assertEqual(cfg.objective_preset,path.stem)
    def test_help_never_probes_cuda(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location("train_cli",ROOT/"scripts/train_breaking_adapter.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(torch.cuda,"is_available",side_effect=AssertionError("GPU probe")), patch("sys.argv",["train","--help"]):
            with self.assertRaises(SystemExit) as cm:module.parse_args()
        self.assertEqual(cm.exception.code,0)

if __name__ == "__main__":unittest.main()
