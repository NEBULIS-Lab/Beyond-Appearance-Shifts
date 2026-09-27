"""Shared, explicit deployment for frozen/core/pres/full carrier runners."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import re
from time import perf_counter
import warnings
import numpy as np

from bas_vla.preserving.pipeline import PreservingPipelineConfig, PreservingPipelineInputs, run_preserving_pipeline
from bas_vla.preserving.backends import build_grounded_backend_bundle_from_env
from bas_vla.preserving.selector import select_preserving_probe_output
from bas_vla.preserving.gate import PreservingGateConfig
from .features import feature_gaps, load_factory, validate_feature_metadata
from .preserving import compute_action_gap_from_chunks

MODES = ("frozen", "core", "pres", "full", "default", "baseline", "breaking")


def normalize_mode(value):
    aliases = {"default": "core", "baseline": "frozen", "breaking": "core"}
    if value not in MODES:
        raise ValueError(f"unknown deployment mode: {value}")
    if value in aliases:
        warnings.warn(f"mode {value!r} is deprecated; use {aliases[value]!r}", FutureWarning, stacklevel=2)
    return aliases.get(value, value)


def add_calibration_arguments(parser):
    parser.add_argument("--mode", "--preserving-mode", dest="mode", choices=MODES, default="frozen")
    parser.add_argument("--residual-checkpoint", type=Path)
    parser.add_argument("--residual-metadata", type=Path)
    parser.add_argument("--residual-vocab", type=Path)
    parser.add_argument("--residual-device", default="cpu")
    parser.add_argument("--residual-scale", type=float, default=1.0, help="Multiplier on checkpoint residual, after its delta_scale")
    parser.add_argument("--residual-action-space", help="Required for legacy metadata without action_space")
    parser.add_argument("--residual-action-layout", choices=("per_step", "flattened_chunk"))
    parser.add_argument("--instruction-provider", help="module:factory returning callable(instruction)->frozen vector")
    parser.add_argument("--feature-provider", help="module:factory for frozen carrier features")
    parser.add_argument("--feature-config", type=Path, help="JSON including exact layers, pooling and normalization")
    parser.add_argument("--gate-config", type=Path, help="JSON fields for PreservingGateConfig")
    parser.add_argument("--preserving-phase-horizon-steps", type=int, default=8)


class ResidualCalibrator:
    def __init__(self, metadata, feature_provider, apply, *, action_space, residual_scale=1.0, action_layout=None):
        self.metadata = dict(metadata)
        if self.metadata.get("action_space") != action_space:
            raise ValueError("residual action_space must explicitly match carrier action convention")
        self.layout = action_layout or metadata.get("action_layout")
        if self.layout not in ("per_step", "flattened_chunk"):
            raise ValueError("residual action_layout required: per_step or flattened_chunk")
        self.feature_provider, self._apply = feature_provider, apply
        self.residual_scale = float(residual_scale)
        if not np.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("residual scale must be finite and nonnegative")

    def apply(self, actions, instruction):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim not in (1, 2) or not np.isfinite(actions).all():
            raise ValueError("actions must be a finite action vector or chunk")
        original_shape = actions.shape
        batch = actions.reshape(1, -1) if self.layout == "flattened_chunk" else actions.reshape(-1, actions.shape[-1])
        features = np.asarray(self.feature_provider(instruction), dtype=np.float32)
        out_dim = int(self.metadata["output_dim"])
        if batch.shape[-1] != out_dim:
            raise ValueError(f"checkpoint output_dim={out_dim} incompatible with action shape {original_shape} ({self.layout})")
        if features.ndim != 1 or features.size + out_dim != int(self.metadata["input_dim"]) or not np.isfinite(features).all():
            raise ValueError("instruction feature dimension/finiteness does not match checkpoint")
        calibrated = np.asarray(self._apply(batch, features), dtype=np.float32)
        if calibrated.shape != batch.shape or not np.isfinite(calibrated).all():
            raise ValueError("residual model returned invalid action shape or values")
        return (batch + self.residual_scale * (calibrated - batch)).reshape(original_shape)

    @classmethod
    def from_args(cls, args, carrier, action_space):
        if not args.residual_checkpoint or not args.residual_metadata:
            raise ValueError("core/full requires --residual-checkpoint and --residual-metadata")
        metadata = json.loads(args.residual_metadata.read_text())
        for key, arg in (("action_space", args.residual_action_space), ("action_layout", args.residual_action_layout)):
            if arg is not None:
                if key in metadata and metadata[key] != arg:
                    raise ValueError(f"CLI {key} contradicts checkpoint metadata")
                metadata[key] = arg
        if args.instruction_provider:
            feature_provider = load_factory(args.instruction_provider, carrier, metadata)
        else:
            if not args.residual_vocab:
                raise ValueError("provide --residual-vocab for binary BoW, or --instruction-provider")
            if metadata.get("feature_type", "binary_bow") not in ("bow", "binary_bow"):
                raise ValueError("checkpoint requires a frozen instruction provider, not BoW")
            vocab = json.loads(args.residual_vocab.read_text())
            if not isinstance(vocab, dict) or sorted(vocab.values()) != list(range(len(vocab))):
                raise ValueError("vocab indices must be contiguous from zero")
            if len(vocab) != int(metadata.get("vocab_size", len(vocab))):
                raise ValueError("vocab_size mismatch")
            def feature_provider(text):
                vector = np.zeros(len(vocab), dtype=np.float32)
                for token in re.findall(r"[a-z0-9]+", text.lower()):
                    if token in vocab:
                        vector[vocab[token]] = 1.0
                return vector
        # Validate conventions before importing or loading the model.
        runtime = cls(metadata, feature_provider, None, action_space=action_space, residual_scale=args.residual_scale)
        from bas_vla.breaking.residual_adapter import load_residual_adapter, apply_residual_adapter
        import torch
        device = torch.device(args.residual_device)
        adapter = load_residual_adapter(args.residual_checkpoint, args.residual_metadata, device)
        runtime._apply = lambda actions, features: apply_residual_adapter(adapter, actions, features, device)
        return runtime


class CalibrationRuntime:
    def __init__(self, mode="frozen", *, calibrator=None, feature_provider=None, config=None, backend=None, synchronize=None):
        self.mode = normalize_mode(mode)
        self.calibrator, self.feature_provider = calibrator, feature_provider
        self.config = config or PreservingPipelineConfig(mode=self.mode)
        gate = self.config.gate_config
        if not (0 <= gate.alpha <= 1) or any(not np.isfinite(value) or value <= 0 for value in
                (gate.kappa_vis, gate.kappa_sem, gate.kappa_act)):
            raise ValueError("gate alpha must be in [0,1] and kappa scales finite and positive")
        if self.mode in ("core", "full") and calibrator is None:
            raise ValueError("core/full requires a loaded residual calibrator")
        if self.mode in ("pres", "full"):
            if feature_provider is None:
                raise ValueError("pres/full requires explicit frozen carrier --feature-provider and --feature-config")
            validate_feature_metadata(feature_provider.metadata)
        self.synchronize = synchronize
        self.backend = backend
        self.chunk_count = 0

    @classmethod
    def from_args(cls, args, carrier, action_space="libero_env"):
        mode = normalize_mode(args.mode)
        calibrator = ResidualCalibrator.from_args(args, carrier, action_space) if mode in ("core", "full") else None
        provider = None
        if mode in ("pres", "full"):
            if not args.feature_provider or not args.feature_config:
                raise ValueError("pres/full requires explicit frozen carrier --feature-provider and --feature-config")
            feature_config = json.loads(args.feature_config.read_text())
            validate_feature_metadata(feature_config)
            provider = load_factory(args.feature_provider, carrier, feature_config)
            validate_feature_metadata(provider.metadata)
            for key in ("mid_layer", "late_layer", "normalization", "pooling"):
                if provider.metadata[key] != feature_config[key]:
                    if hasattr(provider, "close"):
                        provider.close()
                    raise ValueError(f"feature provider metadata contradicts configuration: {key}")
        gate = json.loads(args.gate_config.read_text()) if args.gate_config else {"phase_horizon_steps": args.preserving_phase_horizon_steps}
        return cls(mode, calibrator=calibrator, feature_provider=provider,
                   config=PreservingPipelineConfig(mode=mode, gate_config=PreservingGateConfig(**gate)))

    def predict(self, observation, instruction, infer, step_index=0, *, task_id=None, pass_id=None, warmup=False):
        """Infer a chunk, calibrate it, and return (chunk, complete JSON-safe trace)."""
        def sync():
            if self.synchronize is not None:
                self.synchronize()
        sync()
        started = perf_counter()
        timings = {}
        def infer_once(obs, label):
            tick = perf_counter()
            if self.feature_provider is not None:
                self.feature_provider.before_inference()
            result = infer(obs, instruction)
            actions, internal = result if isinstance(result, tuple) else (result, {})
            features = self.feature_provider.after_inference(obs, instruction) if self.feature_provider is not None else None
            sync()
            timings[label + "_ms"] = (perf_counter() - tick) * 1000
            timings[label + "_carrier"] = internal
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim not in (1, 2) or not actions.size or not np.isfinite(actions).all():
                raise ValueError("carrier returned invalid action chunk")
            return actions, features
        base, base_features = infer_once(observation, "base")
        tick = perf_counter()
        anchor = self.calibrator.apply(base, instruction) if self.mode in ("core", "full") else base.copy()
        sync()
        timings["calibration_ms"] = (perf_counter() - tick) * 1000
        trace = {"step_index": int(step_index), "mode": self.mode, "base_action": base.tolist(),
                 "anchor_action": anchor.tolist(), "probe_enabled": False, "tau": 0.0, "timing": timings}
        if self.calibrator is not None:
            trace["residual_metadata"] = self.calibrator.metadata
            trace["residual_scale"] = self.calibrator.residual_scale
        output = anchor
        if self.mode in ("pres", "full"):
            tick = perf_counter()
            if self.backend is None:
                self.backend = build_grounded_backend_bundle_from_env()
            sync()
            timings["backend_setup_ms"] = (perf_counter() - tick) * 1000
            tick = perf_counter()
            prepared = select_preserving_probe_output(observation, instruction,
                selector_config=self.config.selector_config, grounded_backend_bundle=self.backend,
                grounded_probe_config=self.config.grounded_probe_config, style_probe_config=self.config.style_probe_config,
                style_trigger_config=self.config.style_trigger_config)
            sync()
            timings["probe_prepare_ms"] = (perf_counter() - tick) * 1000
            trace.update(probe_enabled=bool(prepared.enabled), probe_name=prepared.probe_name,
                         backend_name=self.backend.backend_name, feature_metadata=self.feature_provider.metadata,
                         gate_config=asdict(self.config.gate_config))
            if prepared.enabled:
                probe, probe_features = infer_once(prepared.observation, "probe")
                visual, semantic = feature_gaps(base_features, probe_features, self.feature_provider.metadata)
                action_gap = compute_action_gap_from_chunks(base, probe)
                tick = perf_counter()
                result = run_preserving_pipeline(PreservingPipelineInputs(observation=observation, instruction=instruction,
                    step_index=step_index, visual_mid_gap=visual, semantic_late_gap=semantic, action_gap=action_gap,
                    base_action=base, breaking_action=anchor if self.mode == "full" else None, probe_action=probe),
                    replace(self.config, mode=self.mode), grounded_backend_bundle=self.backend, prepared_probe=prepared)
                output = result.deployment.output_action
                sync()
                timings["gate_fusion_ms"] = (perf_counter() - tick) * 1000
                trace.update(visual_gap=visual, semantic_gap=semantic, action_gap=action_gap,
                             gate=asdict(result.gate_scores), tau=result.gate_scores.tau, probe_action=probe.tolist())
        sync()
        timings["total_ms"] = (perf_counter() - started) * 1000
        trace.update(chunk_id=self.chunk_count, warmup=bool(warmup), synchronized=self.synchronize is not None,
                     task_id=task_id, pass_id=pass_id,
                     timing_ms={key[:-3] if key != "total_ms" else "end_to_end": value
                                for key, value in timings.items() if key.endswith("_ms")})
        self.chunk_count += 1
        trace["output_action"] = np.asarray(output).tolist()
        return output, trace
