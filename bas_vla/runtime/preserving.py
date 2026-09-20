from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from bas_vla.preserving import (
    GroundedBackendBundle,
    PreservingPipelineConfig,
    PreservingPipelineInputs,
    PreservingPipelineResult,
    run_preserving_pipeline,
)


def _as_uint8_image(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.uint8).copy()


def _as_float32_array(value: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).copy()


@dataclass(frozen=True)
class PreservingSignalInputs:
    step_index: int
    visual_mid_gap: float = 0.0
    semantic_late_gap: float = 0.0
    action_gap: float = 0.0
    phase_horizon_steps: int | None = None


@dataclass(frozen=True)
class PreservingRuntimeAdapterInputs:
    observation: Mapping[str, np.ndarray]
    instruction: str
    base_action: np.ndarray | list[float] | None = None
    breaking_action: np.ndarray | list[float] | None = None
    probe_action: np.ndarray | list[float] | None = None
    signals: PreservingSignalInputs = field(default_factory=lambda: PreservingSignalInputs(step_index=0))
    target_object_hint: str | None = None
    receptacle_hint: str | None = None


def compute_visual_gap_from_observations(
    base_observation: Mapping[str, np.ndarray],
    probe_observation: Mapping[str, np.ndarray],
) -> float:
    """Legacy pixel-distance diagnostic, not frozen visual-feature evidence."""
    base_image = np.asarray(base_observation["full_image"], dtype=np.float32)
    probe_image = np.asarray(probe_observation["full_image"], dtype=np.float32)
    if base_image.shape != probe_image.shape:
        raise ValueError(
            f"visual-gap observations must have matching image shapes, got {base_image.shape} and {probe_image.shape}"
        )
    return float(np.mean(np.abs(base_image - probe_image)) / 255.0)


def compute_action_gap_from_chunks(
    base_action: np.ndarray | list[float],
    probe_action: np.ndarray | list[float],
) -> float:
    base = _as_float32_array(base_action)
    probe = _as_float32_array(probe_action)
    if base.shape != probe.shape:
        raise ValueError(f"action chunks must share the same shape, got {base.shape} and {probe.shape}")
    if base.ndim == 1:
        return float(np.linalg.norm(base - probe))
    per_step = np.linalg.norm(base - probe, axis=-1)
    return float(np.mean(per_step))


def build_openpi_observation(
    *,
    agentview_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray | list[float],
) -> dict[str, np.ndarray]:
    return {
        "full_image": _as_uint8_image(agentview_image),
        "wrist_image": _as_uint8_image(wrist_image),
        "state": _as_float32_array(state),
    }


def build_openvla_oft_observation(
    *,
    agentview_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray | list[float],
) -> dict[str, np.ndarray]:
    return {
        "full_image": _as_uint8_image(agentview_image),
        "wrist_image": _as_uint8_image(wrist_image),
        "state": _as_float32_array(state),
    }


def build_preserving_pipeline_inputs(
    adapter_inputs: PreservingRuntimeAdapterInputs,
) -> PreservingPipelineInputs:
    signals = adapter_inputs.signals
    return PreservingPipelineInputs(
        observation=adapter_inputs.observation,
        instruction=adapter_inputs.instruction,
        step_index=signals.step_index,
        visual_mid_gap=signals.visual_mid_gap,
        semantic_late_gap=signals.semantic_late_gap,
        action_gap=signals.action_gap,
        base_action=adapter_inputs.base_action,
        breaking_action=adapter_inputs.breaking_action,
        probe_action=adapter_inputs.probe_action,
        phase_horizon_steps=signals.phase_horizon_steps,
        target_object_hint=adapter_inputs.target_object_hint,
        receptacle_hint=adapter_inputs.receptacle_hint,
    )


def run_openpi_preserving_adapter(
    adapter_inputs: PreservingRuntimeAdapterInputs,
    config: PreservingPipelineConfig | None = None,
    *,
    grounded_backend_bundle: GroundedBackendBundle | None = None,
    prepared_probe=None,
) -> PreservingPipelineResult:
    return run_preserving_pipeline(
        build_preserving_pipeline_inputs(adapter_inputs),
        config,
        grounded_backend_bundle=grounded_backend_bundle,
        prepared_probe=prepared_probe,
    )


def run_openvla_oft_preserving_adapter(
    adapter_inputs: PreservingRuntimeAdapterInputs,
    config: PreservingPipelineConfig | None = None,
    *,
    grounded_backend_bundle: GroundedBackendBundle | None = None,
    prepared_probe=None,
) -> PreservingPipelineResult:
    return run_preserving_pipeline(
        build_preserving_pipeline_inputs(adapter_inputs),
        config,
        grounded_backend_bundle=grounded_backend_bundle,
        prepared_probe=prepared_probe,
    )
