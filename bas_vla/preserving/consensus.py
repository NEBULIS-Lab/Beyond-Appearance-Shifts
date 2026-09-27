"""Consensus and image canonicalization utilities for offline diagnostics.

The deployment path is implemented by probe selection, gating, and weak fusion
in ``pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def parse_name_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def apply_gray_world(image: np.ndarray) -> np.ndarray:
    balanced = image.astype(np.float32)
    channel_means = np.mean(balanced, axis=(0, 1), keepdims=True)
    target = float(np.mean(channel_means))
    scale = target / np.clip(channel_means, 1.0, None)
    scale = np.clip(scale, 0.6, 1.8)
    balanced *= scale
    return np.clip(balanced, 0.0, 255.0).astype(np.uint8)


def apply_autocontrast(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    stretched = image.astype(np.float32)
    low = np.percentile(stretched, low_pct, axis=(0, 1), keepdims=True)
    high = np.percentile(stretched, high_pct, axis=(0, 1), keepdims=True)
    denom = np.clip(high - low, 8.0, None)
    stretched = (stretched - low) * (255.0 / denom)
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def apply_exposure_balance(image: np.ndarray, target_mean: float = 118.0) -> np.ndarray:
    adjusted = image.astype(np.float32)
    current_mean = float(np.mean(adjusted))
    scale = target_mean / max(current_mean, 1.0)
    scale = float(np.clip(scale, 0.8, 1.8))
    adjusted *= scale
    adjusted = 255.0 * np.power(np.clip(adjusted / 255.0, 0.0, 1.0), 0.92)
    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


def apply_canonicalizer(image: np.ndarray, canonicalizer: str) -> np.ndarray:
    if canonicalizer == "identity":
        return image
    if canonicalizer == "gray_world":
        return apply_gray_world(image)
    if canonicalizer == "autocontrast":
        return apply_autocontrast(image)
    if canonicalizer == "gray_world_autocontrast":
        return apply_autocontrast(apply_gray_world(image))
    if canonicalizer == "exposure_balance":
        return apply_exposure_balance(image)
    if canonicalizer == "gray_world_exposure":
        return apply_exposure_balance(apply_gray_world(image))
    raise ValueError(f"Unsupported canonicalizer: {canonicalizer}")


@dataclass(frozen=True)
class ConsensusResult:
    actions: list[np.ndarray]
    selected_indices: list[int]
    selected_names: list[str]
    weights: list[float]
    mean_distances: list[float]


def compute_consensus(
    candidate_chunks: list[list[np.ndarray]],
    candidate_names: list[str],
    *,
    top_k: int = 2,
    temperature: float = 10.0,
) -> ConsensusResult:
    if not candidate_chunks:
        raise ValueError("candidate_chunks must not be empty")
    if len(candidate_chunks) != len(candidate_names):
        raise ValueError("candidate_chunks and candidate_names must have the same length")

    if len(candidate_chunks) == 1:
        return ConsensusResult(
            actions=[np.asarray(action, dtype=np.float32) for action in candidate_chunks[0]],
            selected_indices=[0],
            selected_names=[candidate_names[0]],
            weights=[1.0],
            mean_distances=[0.0],
        )

    stacked = np.stack([np.stack(chunk, axis=0) for chunk in candidate_chunks], axis=0)
    flat = stacked.reshape(stacked.shape[0], -1)
    pairwise = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    mean_distances = np.mean(pairwise, axis=1)
    order = np.argsort(mean_distances)

    use_top_k = max(1, min(int(top_k), len(candidate_chunks)))
    support = order[:use_top_k]
    support_distances = mean_distances[support]

    logits = -float(temperature) * (support_distances - np.min(support_distances))
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / np.sum(weights)
    consensus = np.tensordot(weights, stacked[support], axes=(0, 0))

    return ConsensusResult(
        actions=[np.asarray(action, dtype=np.float32) for action in consensus],
        selected_indices=[int(index) for index in support.tolist()],
        selected_names=[candidate_names[int(index)] for index in support.tolist()],
        weights=[float(weight) for weight in weights.tolist()],
        mean_distances=[float(distance) for distance in mean_distances.tolist()],
    )
