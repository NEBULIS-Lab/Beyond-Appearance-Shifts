"""Deterministic mask corruption controls and paired outcome summaries."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def _mask(mask):
    a = np.asarray(mask)
    if a.ndim != 2 or a.size == 0 or not np.isin(a, [0, 1]).all():
        raise ValueError("mask must be a nonempty 2D binary array")
    return a.astype(bool)


def mask_iou(first, second) -> float:
    first, second = _mask(first), _mask(second)
    if first.shape != second.shape:
        raise ValueError("mask shapes differ")
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum()/union) if union else 1.0


def corrupt_mask(mask, *, kind: str, seed: int, target_iou: float | None = None, target_mask=None) -> tuple[np.ndarray, dict]:
    reference = _mask(mask)
    rng = np.random.default_rng(seed)
    out = reference.copy()
    if kind == "iou":
        if target_iou is None or not np.isfinite(target_iou) or not 0 <= target_iou <= 1:
            raise ValueError("iou corruption requires target_iou in [0, 1]")
        indices = np.flatnonzero(reference)
        if not len(indices):
            raise ValueError("IoU corruption needs a nonempty foreground")
        # Foreground dropout gives an auditable target, quantized by mask area.
        keep = int(round(target_iou * len(indices)))
        out[:] = False
        out.flat[rng.choice(indices, size=keep, replace=False)] = True
    elif kind == "missed_target":
        if target_mask is None:
            raise ValueError("missed_target needs an explicit target mask")
        target = _mask(target_mask)
        if target.shape != out.shape or not target.any():
            raise ValueError("target mask must match shape and have foreground")
        out[target] = False
    elif kind == "random":
        out[:] = False
        out.flat[rng.choice(reference.size, size=int(reference.sum()), replace=False)] = True
    elif kind != "normal":
        raise ValueError("kind must be normal, iou, missed_target, or random")
    return out, {"kind": kind, "seed": seed, "requested_iou": target_iou,
                 "achieved_iou": mask_iou(reference, out), "reference_area": int(reference.sum()),
                 "corrupted_area": int(out.sum()),
                 "algorithm": "foreground_dropout" if kind == "iou" else kind}


def summarize_mask_outcomes(rows: list[dict], *, expected_seeds: int = 4, episodes_per_seed: int = 50) -> list[dict]:
    if expected_seeds < 1 or episodes_per_seed < 1:
        raise ValueError("expected seed/episode counts must be positive")
    groups = defaultdict(dict)
    references = {}
    for row in rows:
        for key in ("success", "normal_success"):
            if type(row[key]) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
        group = (str(row["task_id"]), str(row["corruption"]))
        identity = (str(row["seed"]), str(row["episode_id"]))
        reference_id = (group[0], *identity)
        if reference_id in references and references[reference_id] != row["normal_success"]:
            raise ValueError("normal reference outcome differs across corruption levels")
        references[reference_id] = row["normal_success"]
        if identity in groups[group]:
            raise ValueError("duplicate paired mask outcome")
        groups[group][identity] = row
    result = []
    for (task, corruption), paired in sorted(groups.items()):
        values = list(paired.values())
        counts = defaultdict(int)
        for seed, _ in paired:
            counts[seed] += 1
        normal = sum(r["normal_success"] for r in values)
        successes = sum(r["success"] for r in values)
        result.append({"task_id": task, "corruption": corruption, "n": len(values),
                       "successes": successes, "normal_successes": normal,
                       "delta_successes": successes-normal, "delta_percentage_points": 100*(successes-normal)/len(values),
                       "episodes_per_seed": dict(counts),
                       "protocol_complete": len(counts) == expected_seeds and all(n == episodes_per_seed for n in counts.values())})
    return result
