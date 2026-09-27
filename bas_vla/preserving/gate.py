from __future__ import annotations

from dataclasses import dataclass
import math


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class PreservingGateConfig:
    """Concrete gate parameterization for the public BAS-VLA preserving auxiliary.

    The paper specifies the gate as a product of phase, visual, semantic, and
    action-level evidence. This module exposes a simple bounded parameterization
    that matches that structure without hard-coding carrier-specific feature
    extraction logic.
    """

    alpha: float = 0.25
    phase_horizon_steps: int = 8
    phase_start_step: int = 0
    kappa_vis: float = 0.35
    kappa_sem: float = 0.20
    kappa_act: float = 0.10


@dataclass(frozen=True)
class PreservingGateInputs:
    step_index: int
    visual_mid_gap: float
    semantic_late_gap: float
    action_gap: float
    phase_horizon_steps: int | None = None


@dataclass(frozen=True)
class PreservingGateScores:
    tau_phase: float
    g_vis: float
    g_sem: float
    g_act: float
    tau: float


def compute_phase_gate(step_index: int, phase_horizon_steps: int, phase_start_step: int = 0) -> float:
    if phase_horizon_steps <= 0:
        return 0.0
    shifted_step = max(0, int(step_index) - int(phase_start_step))
    return clamp01(1.0 - (shifted_step / float(phase_horizon_steps)))


def compute_visual_gate(visual_mid_gap: float, kappa_vis: float) -> float:
    if kappa_vis <= 0:
        return 0.0
    return clamp01(float(visual_mid_gap) / float(kappa_vis))


def compute_semantic_gate(semantic_late_gap: float, kappa_sem: float) -> float:
    if kappa_sem <= 0:
        return 0.0
    semantic_gap = float(semantic_late_gap)
    scale = float(kappa_sem)
    return clamp01(math.exp(-((semantic_gap * semantic_gap) / (scale * scale))))


def compute_action_gate(action_gap: float, kappa_act: float) -> float:
    if kappa_act <= 0:
        return 0.0
    return clamp01(float(action_gap) / float(kappa_act))


def compute_preserving_gate(
    inputs: PreservingGateInputs,
    config: PreservingGateConfig | None = None,
) -> PreservingGateScores:
    if config is None:
        config = PreservingGateConfig()

    horizon = inputs.phase_horizon_steps if inputs.phase_horizon_steps is not None else config.phase_horizon_steps
    tau_phase = compute_phase_gate(inputs.step_index, horizon, config.phase_start_step)
    g_vis = compute_visual_gate(inputs.visual_mid_gap, config.kappa_vis)
    g_sem = compute_semantic_gate(inputs.semantic_late_gap, config.kappa_sem)
    g_act = compute_action_gate(inputs.action_gap, config.kappa_act)
    tau = clamp01(tau_phase * g_vis * g_sem * g_act)
    return PreservingGateScores(
        tau_phase=tau_phase,
        g_vis=g_vis,
        g_sem=g_sem,
        g_act=g_act,
        tau=tau,
    )
