"""Pure CPU routing/shape tests with fake carriers and fake checkpoint loader."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

from bas_vla.runtime.calibration import CalibrationRuntime, ResidualCalibrator, add_calibration_arguments, normalize_mode
from bas_vla.runtime.features import feature_gaps, NamedModuleFeatures
from bas_vla.preserving.pipeline import PreservingPipelineConfig, PreservingPipelineInputs, run_preserving_pipeline
from bas_vla.preserving.probe import PreservingProbeOutput
from bas_vla.preserving.gate import PreservingGateConfig
from bas_vla.runtime.style import scene_style_properties


class Features:
    metadata = dict(mid_layer="vision.block4", late_layer="vision.block9", pooling="flatten", normalization="none")
    def before_inference(self):
        pass
    def after_inference(self, observation, instruction):
        return {"mid": np.array([observation["mid"]]), "late": np.array([observation["late"]])}


def make_calibrator(layout="per_step", output_dim=2):
    metadata = dict(action_space="libero_env", action_layout=layout, output_dim=output_dim, input_dim=output_dim+1)
    return ResidualCalibrator(metadata, lambda _: np.ones(1), lambda a, f: a+2, action_space="libero_env")


class RuntimeTests(unittest.TestCase):
    def test_modes_fail_closed(self):
        for mode in ("core", "full", "pres"):
            with self.assertRaises(ValueError):
                CalibrationRuntime(mode)
        with self.assertWarns(FutureWarning):
            self.assertEqual(normalize_mode("default"), "core")

    def test_frozen_skips_all_auxiliary_work(self):
        runtime = CalibrationRuntime("frozen")
        with patch("bas_vla.runtime.calibration.build_grounded_backend_bundle_from_env") as backend:
            result, trace = runtime.predict({}, "x", lambda *_: np.ones((2, 2)))
        backend.assert_not_called()
        np.testing.assert_array_equal(result, np.ones((2, 2)))
        self.assertEqual(trace["mode"], "frozen")
        self.assertIn("end_to_end", trace["timing_ms"])

    def test_core_applies_real_calibrator_interface(self):
        runtime = CalibrationRuntime("core", calibrator=make_calibrator())
        result, trace = runtime.predict({}, "x", lambda *_: np.ones((2, 2)))
        np.testing.assert_array_equal(result, np.full((2, 2), 3))
        self.assertEqual(trace["tau"], 0)

    def test_per_step_and_flattened_dimensions(self):
        np.testing.assert_array_equal(make_calibrator().apply(np.zeros(2), "x"), [2, 2])
        with self.assertRaises(ValueError):
            make_calibrator().apply(np.zeros((2, 3)), "x")
        output = make_calibrator("flattened_chunk", 6).apply(np.zeros((2, 3)), "x")
        self.assertEqual(output.shape, (2, 3))
        with self.assertRaises(ValueError):
            make_calibrator("flattened_chunk", 6).apply(np.zeros((3, 3)), "x")

    def test_instruction_dimension_and_nonfinite_rejected(self):
        calibrator = make_calibrator()
        calibrator.feature_provider = lambda _: np.ones(2)
        with self.assertRaises(ValueError):
            calibrator.apply(np.zeros(2), "x")
        with self.assertRaises(ValueError):
            make_calibrator().apply(np.array([float("nan"), 1]), "x")

    def test_feature_semantics_not_constant(self):
        metadata = Features.metadata
        visual, semantic = feature_gaps({"mid": [0], "late": [0]}, {"mid": [1], "late": [2]}, metadata)
        self.assertEqual((visual, semantic), (1, 2))
        with self.assertRaises(ValueError):
            feature_gaps({"mid": [0], "late": [0]}, {"mid": [1], "late": [2]}, {**metadata, "normalization": "l2"})

    def test_full_fuses_calibrated_anchor_but_gate_uses_base(self):
        base_obs = {"mid": 0, "late": 0, "action": 0}
        probe_obs = {"mid": 1, "late": 0, "action": 4}
        prepared = PreservingProbeOutput("fake", probe_obs, {}, True)
        backend = types.SimpleNamespace(backend_name="fake")
        runtime = CalibrationRuntime("full", calibrator=make_calibrator(), feature_provider=Features(),
            config=PreservingPipelineConfig(mode="full", gate_config=PreservingGateConfig(alpha=.25)))
        with patch("bas_vla.runtime.calibration.build_grounded_backend_bundle_from_env", return_value=backend) as build, \
             patch("bas_vla.runtime.calibration.select_preserving_probe_output", return_value=prepared) as select, \
             patch("bas_vla.preserving.pipeline.select_preserving_probe_output", side_effect=AssertionError("probe recomputed")):
            for _ in range(2):
                result, trace = runtime.predict(base_obs, "x", lambda obs, _: np.full((1, 2), obs["action"]))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(select.call_count, 2)
        np.testing.assert_allclose(result, [[2.5, 2.5]])
        self.assertAlmostEqual(trace["action_gap"], np.sqrt(32), places=5)
        self.assertEqual(trace["chunk_id"], 1)

    def test_zero_gate_exact_core_anchor(self):
        prepared = PreservingProbeOutput("fake", {"mid": 1, "late": 0}, {}, True)
        runtime = CalibrationRuntime("full", calibrator=make_calibrator(), feature_provider=Features(),
            backend=types.SimpleNamespace(backend_name="fake"))
        with patch("bas_vla.runtime.calibration.select_preserving_probe_output", return_value=prepared):
            output, trace = runtime.predict({"mid": 0, "late": 0}, "x", lambda *_: np.zeros((1, 2)), step_index=20)
        np.testing.assert_array_equal(output, [[2, 2]])
        self.assertEqual(trace["tau"], 0)

    def test_disabled_probe_does_not_infer_twice(self):
        prepared = PreservingProbeOutput("fake", {}, {}, False)
        runtime = CalibrationRuntime("pres", feature_provider=Features(), backend=types.SimpleNamespace(backend_name="fake"))
        calls = []
        def infer(*_):
            calls.append(1)
            return np.zeros((1, 2))
        with patch("bas_vla.runtime.calibration.select_preserving_probe_output", return_value=prepared):
            _, trace = runtime.predict({"mid": 0, "late": 0}, "x", infer)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("semantic_gap", trace)

    def test_checkpoint_vocab_loaded_and_conventions_checked(self):
        parser = argparse.ArgumentParser()
        add_calibration_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = dict(input_dim=4, output_dim=2, hidden_dim=3, delta_scale=.1, vocab_size=2,
                            action_space="libero_env", action_layout="per_step")
            (root/"metadata.json").write_text(json.dumps(metadata))
            (root/"vocab.json").write_text(json.dumps({"pick": 0, "bowl": 1}))
            args = parser.parse_args(["--mode", "core", "--residual-checkpoint", str(root/"weights.pt"),
                "--residual-metadata", str(root/"metadata.json"), "--residual-vocab", str(root/"vocab.json")])
            seen = []
            fake = types.ModuleType("bas_vla.breaking.residual_adapter")
            fake.load_residual_adapter = lambda *a: seen.append(a) or object()
            fake.apply_residual_adapter = lambda adapter, actions, features, device: actions+features
            torch = types.ModuleType("torch")
            torch.device = lambda value: value
            with patch.dict(sys.modules, {"bas_vla.breaking.residual_adapter": fake, "torch": torch}):
                runtime = CalibrationRuntime.from_args(args, object())
                np.testing.assert_array_equal(runtime.calibrator.apply(np.zeros((1, 2)), "pick pick"), [[1, 0]])
                self.assertEqual(len(seen), 1)
                with self.assertRaises(ValueError):
                    CalibrationRuntime.from_args(args, object(), action_space="wrong")
            self.assertEqual(len(seen), 1)

    def test_hook_names_validated_before_install(self):
        calls = []
        module = types.SimpleNamespace(register_forward_hook=lambda fn: calls.append(fn))
        carrier = types.SimpleNamespace(named_modules=lambda: [("vision.block4", module)])
        with self.assertRaises(ValueError):
            NamedModuleFeatures(carrier, Features.metadata)
        self.assertEqual(calls, [])

    def test_named_hooks_capture_both_real_outputs(self):
        callbacks = {}
        handles = []
        class Tensor:
            def __init__(self, data): self.data = np.array(data)
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
        def register(name):
            def inner(callback):
                callbacks[name] = callback
                handle = types.SimpleNamespace(remove=lambda: handles.append(name))
                return handle
            return inner
        carrier = types.SimpleNamespace(named_modules=lambda: [
            (name, types.SimpleNamespace(register_forward_hook=register(name)))
            for name in ("vision.block4", "vision.block9")])
        provider = NamedModuleFeatures(carrier, {**Features.metadata, "pooling": "mean_tokens"})
        provider.before_inference()
        callbacks["vision.block4"](None, None, Tensor([[1, 2], [3, 4]]))
        callbacks["vision.block9"](None, None, Tensor([[5, 6], [7, 8]]))
        values = provider.after_inference({}, "x")
        np.testing.assert_array_equal(values["mid"], [2, 3])
        np.testing.assert_array_equal(values["late"], [6, 7])
        provider.before_inference()
        with self.assertRaises(RuntimeError):
            provider.after_inference({}, "x")
        provider.close()
        self.assertEqual(len(handles), 2)

    def test_profiling_boundary_and_tags(self):
        boundaries = []
        runtime = CalibrationRuntime("frozen", synchronize=lambda: boundaries.append(1))
        _, trace = runtime.predict({}, "x", lambda *_: np.ones((1, 2)), task_id="bowl", pass_id=2, warmup=True)
        self.assertTrue(trace["synchronized"])
        self.assertTrue(trace["warmup"])
        self.assertEqual(trace["pass_id"], 2)
        self.assertEqual(trace["task_id"], "bowl")
        self.assertGreaterEqual(len(boundaries), 3)

    def test_invalid_gate_scale_rejected(self):
        with self.assertRaises(ValueError):
            CalibrationRuntime("frozen", config=PreservingPipelineConfig(gate_config=PreservingGateConfig(kappa_sem=-1)))

    def test_pipeline_core_never_constructs_probe(self):
        inputs = PreservingPipelineInputs(observation={}, instruction="x", step_index=0,
            visual_mid_gap=0, semantic_late_gap=0, action_gap=0, breaking_action=np.ones(2))
        with patch("bas_vla.preserving.pipeline.build_grounded_backend_bundle_from_env", side_effect=AssertionError("unused")):
            result = run_preserving_pipeline(inputs, PreservingPipelineConfig(mode="core"))
        np.testing.assert_array_equal(result.deployment.output_action, np.ones(2))
        self.assertEqual(result.deployment.mode, "core")

    def test_explicit_style_properties(self):
        self.assertIsNone(scene_style_properties("clean"))
        self.assertEqual(scene_style_properties("style_swap"), {"floor_style": "rustic", "wall_style": "dark-blue"})


if __name__ == "__main__":
    unittest.main()
