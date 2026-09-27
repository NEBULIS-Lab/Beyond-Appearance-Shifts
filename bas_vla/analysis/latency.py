"""Summarize recorded timings without executing a carrier or synchronizing hardware."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def summarize_latency(rows: list[dict], *, min_chunks: int = 200, min_passes: int = 3, allow_unsynchronized: bool = False) -> dict:
    if min_chunks < 1 or min_passes < 1:
        raise ValueError("minimum chunks/passes must be positive")
    groups = defaultdict(list)
    seen = set()
    excluded = 0
    unsynchronized = defaultdict(int)
    for row in rows:
        if type(row.get("warmup")) is not bool or type(row.get("synchronized")) is not bool:
            raise ValueError("warmup and synchronized must be explicit booleans")
        if row["warmup"]:
            excluded += 1
            continue
        if not row["synchronized"] and not allow_unsynchronized:
            raise ValueError("timing rows require synchronized boundaries (CPU completion qualifies)")
        for name in ("task_id", "mode", "pass_id", "chunk_id"):
            if row.get(name) is None or str(row[name]).strip() == "":
                raise ValueError(f"{name} must be explicitly supplied")
        key = tuple(str(row[k]) for k in ("task_id", "mode", "pass_id"))
        unsynchronized[key[:2]] += int(not row["synchronized"])
        identity = (*key, str(row.get("episode_id", "")), str(row["chunk_id"]))
        if identity in seen:
            raise ValueError(f"duplicate timing identity: {identity}")
        seen.add(identity)
        timings = row["timing_ms"]
        if "end_to_end" not in timings:
            raise ValueError("end_to_end must be measured directly, never a component sum")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in timings.values()):
            raise ValueError("timings must be nonnegative finite milliseconds")
        groups[key].append(timings)
    passes, pooled = [], defaultdict(list)
    counts = defaultdict(list)
    def stats(values):
        names = sorted(set().union(*(v.keys() for v in values)))
        return {name: {"n": len(samples := [v[name] for v in values if name in v]),
                       "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95))}
                for name in names}
    for (task, mode, pass_id), values in sorted(groups.items()):
        passes.append({"task_id": task, "mode": mode, "pass_id": pass_id, "chunks": len(values), "metrics": stats(values)})
        pooled[(task, mode)].extend(values)
        counts[(task, mode)].append(len(values))
    summaries = [{"task_id": task, "mode": mode, "passes": len(counts[(task, mode)]),
                  "chunks_per_pass": counts[(task, mode)], "metrics": stats(values),
                  "unsynchronized_chunks": unsynchronized[(task, mode)],
                  "protocol_complete": unsynchronized[(task, mode)] == 0 and len(counts[(task, mode)]) >= min_passes and min(counts[(task, mode)]) >= min_chunks}
                 for (task, mode), values in sorted(pooled.items())]
    return {"unit": "chunk", "percentile_method": "numpy linear; pooled measured chunks, not mean of pass quantiles",
            "min_chunks_per_pass": min_chunks, "min_passes": min_passes,
            "warmup_rows_excluded": excluded, "per_pass": passes, "by_task_mode": summaries}
