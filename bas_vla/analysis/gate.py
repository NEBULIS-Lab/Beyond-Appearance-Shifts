"""Offline gate diagnostics with explicit expert/reference targets."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def probe_better(base_action, probe_action, reference_action) -> bool:
    arrays = [np.asarray(a, dtype=float) for a in (base_action, probe_action, reference_action)]
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays) or arrays[0].size == 0:
        raise ValueError("base, probe, reference must have equal finite nonempty shapes")
    return bool(np.linalg.norm(arrays[1] - arrays[2]) < np.linalg.norm(arrays[0] - arrays[2]))


def _bool(row, key):
    value = row[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a JSON boolean")
    return value


def summarize_gate(rows: list[dict], *, activation_threshold: float = 0.0, unit: str = "chunk") -> dict:
    if not np.isfinite(activation_threshold) or not 0 <= activation_threshold <= 1:
        raise ValueError("activation threshold must be in [0, 1]")
    if unit not in {"chunk", "episode"}:
        raise ValueError("unit must be chunk or episode")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("unit", unit) != unit:
            raise ValueError("do not mix episode and chunk units")
        identity = tuple(str(row[k]) for k in ("task_id", "episode_id", "chunk_id")) if unit == "chunk" else tuple(str(row[k]) for k in ("task_id", "episode_id"))
        identity = (str(row.get("seed", "")), *identity)
        if unit == "chunk" and any(key in row for key in ("oracle_router_success", "full_success")):
            raise ValueError("task retention labels require episode-unit diagnostics to avoid repeated outcomes")
        if identity in seen:
            raise ValueError(f"duplicate diagnostic identity {identity}")
        seen.add(identity)
        changed, shifted = _bool(row, "semantic_changed"), _bool(row, "visual_shift")
        gate = float(row["gate"])
        if not np.isfinite(gate) or not 0 <= gate <= 1:
            raise ValueError("gate must be finite and in [0, 1]")
        target = probe_better(row["base_action"], row["probe_action"], row["reference_action"])
        groups[(changed, shifted)].append((row, gate > activation_threshold, target))
    matrix = []
    for changed in (False, True):
        for shifted in (False, True):
            data = groups[(changed, shifted)]
            n = len(data)
            negatives = sum(not target for _, _, target in data)
            positives = n - negatives
            fp = sum(active and not target for _, active, target in data)
            fn = sum(not active and target for _, active, target in data)
            item = {"semantic_changed": changed, "visual_shift": shifted, "n": n,
                    "target_positive": positives, "target_negative": negatives,
                    "activated": sum(active for _, active, _ in data),
                    "false_activation_count": fp, "false_negative_count": fn,
                    "false_activation_fraction_all": fp/n if n else None,
                    "false_negative_fraction_all": fn/n if n else None,
                    "false_positive_rate": fp/negatives if negatives else None,
                    "false_negative_rate": fn/positives if positives else None}
            # These are supplied outcomes from separate oracle-routed executions.
            oracle = [_bool(r, "oracle_router_success") for r, _, _ in data if "oracle_router_success" in r]
            full = [_bool(r, "full_success") for r, _, _ in data if "full_success" in r]
            for label, values in (("oracle_router", oracle), ("full", full)):
                item[f"{label}_labeled_n"] = len(values)
                item[f"{label}_success_rate"] = sum(values)/len(values) if values else None
            matrix.append(item)
    return {"unit": unit, "activation_rule": f"gate > {activation_threshold}",
            "target": "probe strictly closer than base to expert/reference in L2; ties negative",
            "matrix": matrix}
