"""Flatten supported evaluator timing traces without synthesizing timing metadata."""
from __future__ import annotations


TRACE_KEYS = ("inference_trace", "chunk_trace", "preserving_trace")
TRIPLET_KEYS = {f"{condition}_preserving_trace": condition for condition in ("clean", "control", "break")}


def timing_rows_from_episodes(episodes: list[dict] | dict, *, pass_id: str | None = None) -> list[dict]:
    """Accept episode lists, dual-goal summaries, OpenPI summaries, and OFT triplets."""
    context = {}
    if isinstance(episodes, dict):
        context = episodes
        containers = [key for key in ("episodes", "per_episode", "records") if key in episodes]
        if len(containers) != 1:
            raise ValueError("summary must contain exactly one of episodes, per_episode, or records")
        episodes = episodes[containers[0]]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episode container must be a nonempty list")
    rows = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("episode entries must be objects")
        generic_keys = [key for key in TRACE_KEYS if key in episode]
        triplet_keys = [key for key in TRIPLET_KEYS if key in episode]
        if (generic_keys and triplet_keys) or len(generic_keys) > 1 or not (generic_keys or triplet_keys):
            raise ValueError("episode must supply one inference_trace/chunk_trace/preserving_trace, or named triplet preserving traces")
        episode_id = episode.get("episode_id", episode.get("episode_idx"))
        if episode_id is None:
            raise ValueError("episode needs episode_id or episode_idx")
        seed = episode.get("seed", context.get("seed", ""))
        pair = episode.get("pair_id") or episode.get("case_id") or ""
        for trace_key in generic_keys + triplet_keys:
            traces = episode[trace_key]
            if not isinstance(traces, list) or not traces:
                raise ValueError(f"{trace_key} must be a nonempty list of timing traces")
            condition = TRIPLET_KEYS.get(trace_key, episode.get("instruction_tag") or "")
            for trace in traces:
                if not isinstance(trace, dict):
                    raise ValueError(f"{trace_key} entries must be objects")
                task = trace.get("task_id")
                if task is None:
                    task = next((episode[key] for key in ("task_id", "pair_id", "case_id")
                                 if episode.get(key) is not None), None)
                timing_pass = trace.get("pass_id")
                if timing_pass is None:
                    timing_pass = pass_id
                if task is None or timing_pass is None:
                    raise ValueError("task identity and timing pass must be explicitly supplied")
                for key in ("mode", "chunk_id", "warmup", "synchronized", "timing_ms"):
                    if key not in trace:
                        raise ValueError(f"{trace_key} missing {key}; legacy traces without timing metadata cannot be upgraded")
                row = {key: trace[key] for key in ("mode", "chunk_id", "warmup", "synchronized", "timing_ms")}
                # Include condition and pair so independently indexed triplets stay distinct.
                identity = f"{seed}:{episode_id}"
                if pair or condition:
                    identity += f":{pair}:{condition}"
                row.update(task_id=task, pass_id=timing_pass, episode_id=identity,
                           condition=condition, source_trace=trace_key)
                rows.append(row)
    return rows
