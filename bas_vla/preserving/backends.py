from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .grounded_probe import (
    GroundedProbeConfig,
    GroundingBackend,
    GroundingBox,
    SegmentationBackend,
)


@dataclass(frozen=True)
class GroundedBackendBundle:
    grounding_backend: GroundingBackend
    segmentation_backend: SegmentationBackend
    available: bool
    backend_name: str
    note: str


@dataclass(frozen=True)
class TransformersGroundedBackendConfig:
    grounding_dino_model_id: str
    sam2_model_id: str
    device: str = "cpu"


class UnavailableGroundingBackend(GroundingBackend):
    """Placeholder grounding backend used when no real runtime is configured."""

    def __init__(self, note: str) -> None:
        self.note = note

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        queries: Sequence[str],
        config: GroundedProbeConfig,
    ) -> list[GroundingBox]:
        return []


class UnavailableSegmentationBackend(SegmentationBackend):
    """Placeholder segmentation backend used when no real runtime is configured."""

    def __init__(self, note: str) -> None:
        self.note = note

    def refine(
        self,
        image: np.ndarray,
        boxes: Sequence[GroundingBox],
        config: GroundedProbeConfig,
    ) -> list[np.ndarray]:
        height, width = image.shape[:2]
        return [np.zeros((height, width), dtype=np.float32) for _ in boxes]


def _load_transformers_symbols() -> tuple[Any, Any, Any, Any]:
    try:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )
    except Exception as exc:
        raise RuntimeError(
            "Transformers-based grounded preserving backends require the `transformers` package "
            "with Grounding DINO and SAM2 support. Install the repository `requirements.txt` first."
        ) from exc
    return AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor


def _select_torch_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    return batch


def _normalize_query_text(queries: Sequence[str]) -> str:
    normalized: list[str] = []
    for query in queries:
        item = str(query).strip()
        if not item:
            continue
        if item.startswith("instruction::"):
            item = item.split("::", 1)[1].strip()
        if item and not item.endswith("."):
            item = f"{item}."
        if item:
            normalized.append(item)
    return " ".join(normalized)


class TransformersGroundingDinoBackend(GroundingBackend):
    """Grounding DINO backend backed by Hugging Face Transformers."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = _select_torch_device(device)
        self._model = None
        self._processor = None

    def _lazy_init(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        AutoModelForZeroShotObjectDetection, AutoProcessor, _, _ = _load_transformers_symbols()
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        self._processor = processor
        self._model = model

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        queries: Sequence[str],
        config: GroundedProbeConfig,
    ) -> list[GroundingBox]:
        self._lazy_init()
        assert self._model is not None and self._processor is not None

        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        text = _normalize_query_text(queries)
        if not text:
            return []

        inputs = self._processor(images=pil_image, text=text, return_tensors="pt")
        inputs = _move_batch_to_device(inputs, self.device)
        outputs = self._model(**inputs)

        target_sizes = [pil_image.size[::-1]]
        if hasattr(self._processor, "post_process_grounded_object_detection"):
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids"),
                threshold=float(config.box_threshold),
                text_threshold=float(config.text_threshold),
                target_sizes=target_sizes,
            )
        elif hasattr(self._processor, "image_processor") and hasattr(
            self._processor.image_processor, "post_process_grounded_object_detection"
        ):
            results = self._processor.image_processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids"),
                threshold=float(config.box_threshold),
                text_threshold=float(config.text_threshold),
                target_sizes=target_sizes,
            )
        else:
            raise RuntimeError(
                "The installed transformers package does not expose Grounding DINO post-processing helpers."
            )

        result = results[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        output: list[GroundingBox] = []
        for label, box, score in zip(labels, boxes, scores):
            output.append(
                GroundingBox(
                    label=str(label),
                    xyxy=tuple(float(v) for v in box),
                    score=float(score),
                )
            )
        return output


class TransformersSam2Backend(SegmentationBackend):
    """SAM2 backend backed by Hugging Face Transformers."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = _select_torch_device(device)
        self._model = None
        self._processor = None

    def _lazy_init(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        _, _, Sam2Model, Sam2Processor = _load_transformers_symbols()
        processor = Sam2Processor.from_pretrained(self.model_id)
        model = Sam2Model.from_pretrained(self.model_id)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        self._processor = processor
        self._model = model

    def refine(
        self,
        image: np.ndarray,
        boxes: Sequence[GroundingBox],
        config: GroundedProbeConfig,
    ) -> list[np.ndarray]:
        self._lazy_init()
        assert self._model is not None and self._processor is not None
        if not boxes:
            return []

        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        box_values = [[list(map(float, box.xyxy)) for box in boxes]]
        inputs = self._processor(images=pil_image, input_boxes=box_values, return_tensors="pt")
        inputs = _move_batch_to_device(inputs, self.device)
        outputs = self._model(**inputs)

        if not hasattr(self._processor, "post_process_masks"):
            raise RuntimeError("The installed transformers package does not expose SAM2 post_process_masks.")

        processed = self._processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs.get("original_sizes").detach().cpu(),
            inputs.get("reshaped_input_sizes").detach().cpu(),
        )
        masks_array = np.asarray(processed[0])
        if masks_array.ndim == 4 and masks_array.shape[1] == 1:
            masks_array = masks_array[:, 0]
        if masks_array.ndim == 2:
            masks_array = masks_array[None, ...]
        return [np.asarray(mask, dtype=np.float32) for mask in masks_array]


def build_unavailable_backend_bundle(
    note: str = (
        "Grounded probe backends are not configured. Set BAS_GROUNDING_DINO_MODEL_ID and "
        "BAS_SAM2_MODEL_ID, then install `requirements.txt` to enable Grounding DINO + SAM2."
    ),
) -> GroundedBackendBundle:
    return GroundedBackendBundle(
        grounding_backend=UnavailableGroundingBackend(note),
        segmentation_backend=UnavailableSegmentationBackend(note),
        available=False,
        backend_name="unavailable",
        note=note,
    )


def build_transformers_backend_bundle(
    config: TransformersGroundedBackendConfig,
) -> GroundedBackendBundle:
    _load_transformers_symbols()
    grounding_backend = TransformersGroundingDinoBackend(
        model_id=config.grounding_dino_model_id,
        device=config.device,
    )
    segmentation_backend = TransformersSam2Backend(
        model_id=config.sam2_model_id,
        device=config.device,
    )
    return GroundedBackendBundle(
        grounding_backend=grounding_backend,
        segmentation_backend=segmentation_backend,
        available=True,
        backend_name="transformers_groundingdino_sam2",
        note=(
            f"grounding_dino={config.grounding_dino_model_id}, "
            f"sam2={config.sam2_model_id}, device={config.device}"
        ),
    )


def build_grounded_backend_bundle_from_env() -> GroundedBackendBundle:
    grounding_model = os.environ.get("BAS_GROUNDING_DINO_MODEL_ID", "").strip()
    sam2_model = os.environ.get("BAS_SAM2_MODEL_ID", "").strip()
    device = os.environ.get("BAS_GROUNDED_DEVICE", "auto").strip() or "auto"

    if not grounding_model and not sam2_model:
        return build_unavailable_backend_bundle()
    if not grounding_model or not sam2_model:
        raise RuntimeError(
            "Both BAS_GROUNDING_DINO_MODEL_ID and BAS_SAM2_MODEL_ID must be set to enable grounded preserving backends."
        )

    return build_transformers_backend_bundle(
        TransformersGroundedBackendConfig(
            grounding_dino_model_id=grounding_model,
            sam2_model_id=sam2_model,
            device=device,
        )
    )
