from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .fusion import fuse_full_action, fuse_preserving_action
from .gate import PreservingGateConfig, PreservingGateScores


PreservingMode = Literal["default", "pres", "full"]


@dataclass(frozen=True)
class PreservingDeploymentResult:
    mode: PreservingMode
    output_action: np.ndarray
    anchor_action: np.ndarray
    probe_action: np.ndarray | None
    gate_scores: PreservingGateScores | None


def deploy_default_action(
    breaking_action: np.ndarray,
) -> PreservingDeploymentResult:
    action = np.asarray(breaking_action, dtype=np.float32)
    return PreservingDeploymentResult(
        mode="default",
        output_action=action,
        anchor_action=action,
        probe_action=None,
        gate_scores=None,
    )


def deploy_preserving_action(
    base_action: np.ndarray,
    probe_action: np.ndarray,
    gate_scores: PreservingGateScores,
    config: PreservingGateConfig | None = None,
) -> PreservingDeploymentResult:
    if config is None:
        config = PreservingGateConfig()
    base = np.asarray(base_action, dtype=np.float32)
    probe = np.asarray(probe_action, dtype=np.float32)
    return PreservingDeploymentResult(
        mode="pres",
        output_action=fuse_preserving_action(base, probe, gate_scores, config),
        anchor_action=base,
        probe_action=probe,
        gate_scores=gate_scores,
    )


def deploy_full_action(
    breaking_action: np.ndarray,
    probe_action: np.ndarray,
    gate_scores: PreservingGateScores,
    config: PreservingGateConfig | None = None,
) -> PreservingDeploymentResult:
    if config is None:
        config = PreservingGateConfig()
    breaking = np.asarray(breaking_action, dtype=np.float32)
    probe = np.asarray(probe_action, dtype=np.float32)
    return PreservingDeploymentResult(
        mode="full",
        output_action=fuse_full_action(breaking, probe, gate_scores, config),
        anchor_action=breaking,
        probe_action=probe,
        gate_scores=gate_scores,
    )
