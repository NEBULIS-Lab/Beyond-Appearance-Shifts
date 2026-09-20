from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapSummary:
    mean: float
    ci_low: float
    ci_high: float
    num_samples: int
    num_bootstrap: int


def _bootstrap_indices(size: int, num_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, size, size=(num_bootstrap, size))


def _quantiles(confidence: float) -> tuple[float, float]:
    alpha = max(0.0, min(1.0, 1.0 - confidence / 100.0))
    return alpha / 2.0, 1.0 - alpha / 2.0


def bootstrap_mean(
    values: list[float] | np.ndarray,
    *,
    num_bootstrap: int = 2000,
    confidence: float = 95.0,
    seed: int = 7,
) -> BootstrapSummary:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap_mean expects a non-empty 1D array")
    boot = array[_bootstrap_indices(array.size, num_bootstrap, seed)].mean(axis=1)
    q_low, q_high = _quantiles(confidence)
    return BootstrapSummary(
        mean=float(array.mean()),
        ci_low=float(np.quantile(boot, q_low)),
        ci_high=float(np.quantile(boot, q_high)),
        num_samples=int(array.size),
        num_bootstrap=int(num_bootstrap),
    )


def bootstrap_rate(
    values: list[int] | list[bool] | np.ndarray,
    *,
    num_bootstrap: int = 2000,
    confidence: float = 95.0,
    seed: int = 7,
) -> BootstrapSummary:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap_rate expects a non-empty 1D array")
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError("bootstrap_rate expects binary values")
    return bootstrap_mean(array, num_bootstrap=num_bootstrap, confidence=confidence, seed=seed)


def bootstrap_paired_difference(
    reference: list[float] | np.ndarray,
    variant: list[float] | np.ndarray,
    *,
    num_bootstrap: int = 2000,
    confidence: float = 95.0,
    seed: int = 7,
) -> BootstrapSummary:
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(variant, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.size != b.size:
        raise ValueError("bootstrap_paired_difference expects equal-length non-empty 1D arrays")
    diff = b - a
    return bootstrap_mean(diff, num_bootstrap=num_bootstrap, confidence=confidence, seed=seed)
