from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GroundedProbeConfig:
    """Static grounded-mask probe skeleton for the BAS-VLA preserving auxiliary.

    This file matches the paper-facing preserving design but intentionally avoids
    bundling third-party Grounding DINO or SAM 2 code into the public repository.
    Runtime use is expected to happen through dependency injection of compatible
    grounding and mask-refinement backends.
    """

    box_threshold: float = 0.25
    text_threshold: float = 0.20
    mask_threshold: float = 0.50
    background_keep_color_ratio: float = 0.20
    background_brightness_scale: float = 0.72
    include_robot_arm: bool = True
    include_gripper: bool = True
    include_target_object: bool = True
    include_receptacle: bool = True


@dataclass(frozen=True)
class GroundingBox:
    label: str
    xyxy: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class GroundingResult:
    queries: list[str]
    boxes: list[GroundingBox]


@dataclass(frozen=True)
class MaskRefinementResult:
    masks: list[np.ndarray]
    merged_mask: np.ndarray


@dataclass(frozen=True)
class GroundedProbeOutput:
    probe_name: str
    observation: dict[str, np.ndarray]
    diagnostics: dict[str, float | bool | list[str]]
    enabled: bool
    grounding: GroundingResult
    mask_refinement: MaskRefinementResult


class GroundingBackend(Protocol):
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        queries: Sequence[str],
        config: GroundedProbeConfig,
    ) -> Sequence[GroundingBox]: ...


class SegmentationBackend(Protocol):
    def refine(
        self,
        image: np.ndarray,
        boxes: Sequence[GroundingBox],
        config: GroundedProbeConfig,
    ) -> Sequence[np.ndarray]: ...


def build_grounding_queries(
    instruction: str,
    config: GroundedProbeConfig | None = None,
    *,
    target_object_hint: str | None = None,
    receptacle_hint: str | None = None,
) -> list[str]:
    if config is None:
        config = GroundedProbeConfig()

    queries: list[str] = []
    if config.include_robot_arm:
        queries.append("robot arm")
    if config.include_gripper:
        queries.append("gripper")
    if config.include_target_object:
        queries.append(target_object_hint.strip() if target_object_hint else "target object")
    if config.include_receptacle:
        queries.append(receptacle_hint.strip() if receptacle_hint else "receptacle")

    instruction_marker = instruction.strip()
    if instruction_marker:
        queries.append(f"instruction::{instruction_marker}")
    return queries


def merge_foreground_masks(
    image_shape: tuple[int, int],
    masks: Sequence[np.ndarray],
    threshold: float = 0.50,
) -> np.ndarray:
    height, width = image_shape
    if not masks:
        return np.zeros((height, width), dtype=bool)

    stack = []
    for mask in masks:
        mask_arr = np.asarray(mask, dtype=np.float32)
        if mask_arr.shape != (height, width):
            raise ValueError(f"mask shape {mask_arr.shape} does not match image shape {(height, width)}")
        stack.append(mask_arr)

    merged = np.max(np.stack(stack, axis=0), axis=0)
    return (merged >= float(threshold)).astype(bool)


def attenuate_background(
    image: np.ndarray,
    *,
    keep_color_ratio: float,
    brightness_scale: float,
) -> np.ndarray:
    pil_image = Image.fromarray(image).convert("RGB")
    gray = pil_image.convert("L").convert("RGB")
    blended = Image.blend(gray, pil_image, alpha=float(keep_color_ratio))
    arr = np.asarray(blended, dtype=np.float32)
    arr *= float(brightness_scale)
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def apply_mask_overlay(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    keep_color_ratio: float,
    brightness_scale: float,
) -> np.ndarray:
    foreground = np.asarray(image, dtype=np.uint8)
    if foreground_mask.dtype != bool:
        foreground_mask = np.asarray(foreground_mask, dtype=bool)
    background = attenuate_background(
        foreground,
        keep_color_ratio=keep_color_ratio,
        brightness_scale=brightness_scale,
    )
    mask_3 = foreground_mask[..., None]
    return np.where(mask_3, foreground, background).astype(np.uint8)


def build_grounded_probe_output(
    observation: Mapping[str, np.ndarray],
    instruction: str,
    *,
    grounding_backend: GroundingBackend,
    segmentation_backend: SegmentationBackend,
    config: GroundedProbeConfig | None = None,
    target_object_hint: str | None = None,
    receptacle_hint: str | None = None,
) -> GroundedProbeOutput:
    if config is None:
        config = GroundedProbeConfig()

    full_image = np.asarray(observation["full_image"], dtype=np.uint8).copy()
    wrist_image = np.asarray(observation["wrist_image"], dtype=np.uint8).copy()
    state = np.asarray(observation["state"], dtype=np.float32).copy()

    queries = build_grounding_queries(
        instruction,
        config,
        target_object_hint=target_object_hint,
        receptacle_hint=receptacle_hint,
    )
    boxes = list(grounding_backend.predict(full_image, instruction, queries, config))
    masks = [
        np.asarray(mask, dtype=np.float32)
        for mask in segmentation_backend.refine(full_image, boxes, config)
    ]
    merged_mask = merge_foreground_masks(full_image.shape[:2], masks, threshold=config.mask_threshold)
    probe_full_image = apply_mask_overlay(
        full_image,
        merged_mask,
        keep_color_ratio=config.background_keep_color_ratio,
        brightness_scale=config.background_brightness_scale,
    )

    foreground_fraction = float(merged_mask.mean()) if merged_mask.size else 0.0
    diagnostics: dict[str, float | bool | list[str]] = {
        "foreground_fraction": foreground_fraction,
        "num_boxes": float(len(boxes)),
        "num_masks": float(len(masks)),
        "mask_threshold": float(config.mask_threshold),
        "grounded_probe_active": bool(len(boxes) > 0 and foreground_fraction > 0.0),
        "queries": list(queries),
        **{key: float(value) if isinstance(value, (int, float, bool)) else value for key, value in asdict(config).items()},
    }

    return GroundedProbeOutput(
        probe_name="grounded_style_probe",
        observation={
            "full_image": probe_full_image,
            "wrist_image": wrist_image,
            "state": state,
        },
        diagnostics=diagnostics,
        enabled=bool(diagnostics["grounded_probe_active"]),
        grounding=GroundingResult(queries=list(queries), boxes=list(boxes)),
        mask_refinement=MaskRefinementResult(masks=masks, merged_mask=merged_mask),
    )
