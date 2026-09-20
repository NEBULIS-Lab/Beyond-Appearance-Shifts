from __future__ import annotations

import numpy as np

from .gate import PreservingGateConfig, PreservingGateScores


def _as_float32_array(value: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def weak_fuse_actions(
    anchor_action: np.ndarray | list[float],
    probe_action: np.ndarray | list[float],
    tau: float,
    alpha: float,
) -> np.ndarray:
    anchor = _as_float32_array(anchor_action)
    probe = _as_float32_array(probe_action)
    weight = float(alpha) * float(tau)
    return ((1.0 - weight) * anchor + weight * probe).astype(np.float32)


def fuse_preserving_action(
    base_action: np.ndarray | list[float],
    probe_action: np.ndarray | list[float],
    gate_scores: PreservingGateScores,
    config: PreservingGateConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = PreservingGateConfig()
    return weak_fuse_actions(base_action, probe_action, gate_scores.tau, config.alpha)


def fuse_full_action(
    breaking_action: np.ndarray | list[float],
    probe_action: np.ndarray | list[float],
    gate_scores: PreservingGateScores,
    config: PreservingGateConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = PreservingGateConfig()
    return weak_fuse_actions(breaking_action, probe_action, gate_scores.tau, config.alpha)
