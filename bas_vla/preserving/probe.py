from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StyleProbeConfig:
    """Default style-oriented preserving probe used by the public BAS-VLA sidecar.

    The current paper narrative uses a generic notation ``phi_sty`` for the probe
    transform. In the public codebase we expose a concrete, validated
    style-oriented instantiation rather than a universal nuisance transform.
    """

    top_mask_ratio: float = 0.28
    side_mask_ratio: float = 0.0
    keep_color_ratio: float = 0.25


@dataclass(frozen=True)
class StyleTriggerConfig:
    """Heuristic trigger thresholds for the default style-oriented probe."""

    top_saturation_threshold: float = 0.08
    top_blue_dominance_threshold: float = 0.015


@dataclass(frozen=True)
class PreservingProbeOutput:
    """Probe output used by the preserving auxiliary."""

    probe_name: str
    observation: dict[str, np.ndarray]
    diagnostics: dict[str, float | bool]
    enabled: bool


def _desaturate_image(image: np.ndarray, keep_color_ratio: float) -> np.ndarray:
    pil_image = Image.fromarray(image).convert("RGB")
    gray = pil_image.convert("L").convert("RGB")
    blended = Image.blend(gray, pil_image, alpha=keep_color_ratio)
    return np.asarray(blended, dtype=np.uint8).copy()


def compute_style_probe_statistics(image: np.ndarray, top_mask_ratio: float) -> tuple[float, float]:
    normalized = image.astype(np.float32) / 255.0
    height = normalized.shape[0]
    top_pixels = max(1, int(round(height * top_mask_ratio)))
    top_region = normalized[:top_pixels]

    max_rgb = top_region.max(axis=2)
    min_rgb = top_region.min(axis=2)
    saturation = np.divide(
        max_rgb - min_rgb,
        np.maximum(max_rgb, 1e-6),
        out=np.zeros_like(max_rgb),
        where=max_rgb > 1e-6,
    )
    rgb_mean = top_region.reshape(-1, 3).mean(axis=0)
    blue_dominance = float(rgb_mean[2] - 0.5 * (rgb_mean[0] + rgb_mean[1]))
    return float(saturation.mean()), blue_dominance


def compute_style_probe_diagnostics(
    image: np.ndarray,
    probe_cfg: StyleProbeConfig,
    trigger_cfg: StyleTriggerConfig | None = None,
) -> dict[str, float | bool]:
    if trigger_cfg is None:
        trigger_cfg = StyleTriggerConfig()

    top_saturation, top_blue_dominance = compute_style_probe_statistics(image, probe_cfg.top_mask_ratio)
    return {
        "top_saturation": float(top_saturation),
        "top_blue_dominance": float(top_blue_dominance),
        "style_trigger_active": bool(
            top_saturation >= trigger_cfg.top_saturation_threshold
            and top_blue_dominance >= trigger_cfg.top_blue_dominance_threshold
        ),
    }


def build_style_probe_view(image: np.ndarray, probe_cfg: StyleProbeConfig) -> np.ndarray:
    transformed = _desaturate_image(image, probe_cfg.keep_color_ratio)
    height, width = transformed.shape[:2]
    top_pixels = max(1, int(round(height * probe_cfg.top_mask_ratio)))
    side_pixels = 0 if probe_cfg.side_mask_ratio <= 0 else max(1, int(round(width * probe_cfg.side_mask_ratio)))

    workspace = transformed[int(0.35 * height) : int(0.95 * height), int(0.10 * width) : int(0.90 * width)]
    fill_color = workspace.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    transformed[:top_pixels, :] = fill_color
    if side_pixels > 0:
        transformed[:, :side_pixels] = fill_color
        transformed[:, width - side_pixels :] = fill_color
    return transformed


def build_style_probe_output(
    observation: Mapping[str, np.ndarray],
    probe_cfg: StyleProbeConfig | None = None,
    trigger_cfg: StyleTriggerConfig | None = None,
) -> PreservingProbeOutput:
    if probe_cfg is None:
        probe_cfg = StyleProbeConfig()
    if trigger_cfg is None:
        trigger_cfg = StyleTriggerConfig()

    full_image = np.asarray(observation["full_image"], dtype=np.uint8).copy()
    wrist_image = np.asarray(observation["wrist_image"], dtype=np.uint8).copy()
    state = np.asarray(observation["state"], dtype=np.float32).copy()

    diagnostics = compute_style_probe_diagnostics(full_image, probe_cfg, trigger_cfg)
    probe_observation = {
        "full_image": build_style_probe_view(full_image, probe_cfg),
        "wrist_image": wrist_image,
        "state": state,
    }
    return PreservingProbeOutput(
        probe_name="style_probe",
        observation=probe_observation,
        diagnostics={
            **diagnostics,
            **{key: float(value) for key, value in asdict(probe_cfg).items()},
            **{key: float(value) for key, value in asdict(trigger_cfg).items()},
        },
        enabled=bool(diagnostics["style_trigger_active"]),
    )
