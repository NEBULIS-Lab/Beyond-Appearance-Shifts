from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .backends import GroundedBackendBundle, build_grounded_backend_bundle_from_env
from .grounded_probe import GroundedProbeConfig, GroundedProbeOutput, build_grounded_probe_output
from .probe import (
    PreservingProbeOutput,
    StyleProbeConfig,
    StyleTriggerConfig,
    build_style_probe_output,
)


@dataclass(frozen=True)
class PreservingProbeSelectorConfig:
    prefer_grounded_probe: bool = True
    allow_style_fallback: bool = True


def select_preserving_probe_output(
    observation: Mapping[str, np.ndarray],
    instruction: str,
    *,
    selector_config: PreservingProbeSelectorConfig | None = None,
    grounded_backend_bundle: GroundedBackendBundle | None = None,
    grounded_probe_config: GroundedProbeConfig | None = None,
    style_probe_config: StyleProbeConfig | None = None,
    style_trigger_config: StyleTriggerConfig | None = None,
    target_object_hint: str | None = None,
    receptacle_hint: str | None = None,
) -> GroundedProbeOutput | PreservingProbeOutput:
    if selector_config is None:
        selector_config = PreservingProbeSelectorConfig()
    if grounded_backend_bundle is None:
        grounded_backend_bundle = build_grounded_backend_bundle_from_env()

    if selector_config.prefer_grounded_probe and grounded_backend_bundle.available:
        return build_grounded_probe_output(
            observation,
            instruction,
            grounding_backend=grounded_backend_bundle.grounding_backend,
            segmentation_backend=grounded_backend_bundle.segmentation_backend,
            config=grounded_probe_config,
            target_object_hint=target_object_hint,
            receptacle_hint=receptacle_hint,
        )

    if selector_config.allow_style_fallback:
        return build_style_probe_output(
            observation,
            probe_cfg=style_probe_config,
            trigger_cfg=style_trigger_config,
        )

    return build_grounded_probe_output(
        observation,
        instruction,
        grounding_backend=grounded_backend_bundle.grounding_backend,
        segmentation_backend=grounded_backend_bundle.segmentation_backend,
        config=grounded_probe_config,
        target_object_hint=target_object_hint,
        receptacle_hint=receptacle_hint,
    )
