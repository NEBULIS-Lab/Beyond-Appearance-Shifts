"""Explicit frozen carrier features; no guessed layers or pixel substitutes."""
from __future__ import annotations

import importlib
import numpy as np


def load_factory(spec, carrier, config):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)(carrier=carrier, config=config)


def validate_feature_metadata(metadata):
    for key in ("mid_layer", "late_layer", "normalization", "pooling"):
        if not metadata.get(key):
            raise ValueError(f"feature provider metadata requires {key}")
    if metadata["normalization"] not in ("none", "l2"):
        raise ValueError("feature normalization must be none or l2")


def feature_gaps(base, probe, metadata):
    validate_feature_metadata(metadata)
    gaps = []
    for key in ("mid", "late"):
        a, b = (np.asarray(x[key], dtype=np.float32).reshape(-1) for x in (base, probe))
        if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid or mismatched {key} feature arrays")
        if metadata["normalization"] == "l2":
            an, bn = np.linalg.norm(a), np.linalg.norm(b)
            if an == 0 or bn == 0:
                raise ValueError("cannot L2-normalize zero features")
            a, b = a / an, b / bn
        gaps.append(float(np.linalg.norm(a - b)))
    return tuple(gaps)


class NamedModuleFeatures:
    """Capture explicitly named torch modules during existing carrier forwards.

    Config requires mid_layer, late_layer, normalization and pooling. Pooling is
    flatten or mean_tokens (last dimension is channels). Optional output_index
    selects a tuple output. Repeated module invocations are concatenated in order.
    """
    def __init__(self, carrier, config):
        self.metadata = dict(config)
        validate_feature_metadata(self.metadata)
        if config["pooling"] not in ("flatten", "mean_tokens"):
            raise ValueError("pooling must be flatten or mean_tokens")
        modules = dict(carrier.named_modules())
        for key in ("mid", "late"):
            if config[f"{key}_layer"] not in modules:
                raise ValueError(f"carrier has no module named {config[f'{key}_layer']}")
        self.handles = []
        self.values = {}
        for key in ("mid", "late"):
            name = config[f"{key}_layer"]
            if name not in modules:
                raise ValueError(f"carrier has no module named {name}")
            self.handles.append(modules[name].register_forward_hook(self._hook(key)))

    def _hook(self, key):
        def capture(_module, _inputs, output):
            if "output_index" in self.metadata:
                output = output[self.metadata["output_index"]]
            if not hasattr(output, "detach"):
                raise ValueError("hook output must be a tensor; configure output_index for tuples")
            value = output.detach().float().cpu().numpy()
            if self.metadata["pooling"] == "mean_tokens":
                value = value.reshape(-1, value.shape[-1]).mean(axis=0)
            self.values.setdefault(key, []).append(value.reshape(-1).copy())
        return capture

    def before_inference(self):
        self.values = {}

    def after_inference(self, observation, instruction):
        if any(key not in self.values for key in ("mid", "late")):
            raise RuntimeError("selected carrier layers did not execute during inference")
        return {key: np.concatenate(self.values[key]) for key in ("mid", "late")}

    def close(self):
        for handle in self.handles:
            handle.remove()


def named_module_features(*, carrier, config):
    return NamedModuleFeatures(carrier, config)
