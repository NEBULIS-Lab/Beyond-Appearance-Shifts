from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_KEYS = (
    "case_id",
    "family",
    "split",
    "clean_instruction",
    "control_instruction",
    "break_instruction",
    "clean_base_action",
    "control_base_action",
    "break_base_action",
    "expert_action",
)


def load_cached_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("records path must point to a JSON list")
    return payload


def _vector_dim(record: dict[str, Any], key: str) -> int:
    value = record.get(key, [])
    if not isinstance(value, list):
        return -1
    return len(value)


def summarize_cached_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    split_family_counter: dict[str, Counter[str]] = defaultdict(Counter)
    missing_keys: Counter[str] = Counter()
    action_dims: dict[str, Counter[int]] = {
        "clean_base_action": Counter(),
        "control_base_action": Counter(),
        "break_base_action": Counter(),
        "expert_action": Counter(),
    }

    for record in records:
        split = str(record.get("split", "missing"))
        family = str(record.get("family", "missing"))
        split_counter[split] += 1
        family_counter[family] += 1
        split_family_counter[split][family] += 1

        for key in REQUIRED_KEYS:
            if key not in record:
                missing_keys[key] += 1

        for key in action_dims:
            action_dims[key][_vector_dim(record, key)] += 1

    return {
        "num_records": len(records),
        "splits": dict(sorted(split_counter.items())),
        "families": dict(sorted(family_counter.items())),
        "split_family_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(split_family_counter.items())
        },
        "missing_required_key_counts": dict(sorted(missing_keys.items())),
        "action_dims": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(action_dims.items())
        },
    }


ACTION_KEYS = ("clean_base_action", "control_base_action", "break_base_action", "expert_action")


def validate_cached_records(records: list[dict[str, Any]], *, require_splits: bool = True) -> int:
    """Validate finite, consistently sized single-action triplets. Return action dimension."""
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty JSON list")
    dimensions: set[int] = set()
    splits: set[str] = set()
    seen: set[str] = set()
    source_splits: dict[tuple[str, str], str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        missing = set(REQUIRED_KEYS) - record.keys()
        if missing:
            raise ValueError(f"record {index}: missing {sorted(missing)}")
        for key in REQUIRED_KEYS[:6]:
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"record {index}: {key} must be a nonempty string")
        if not isinstance(record["break_instruction"], str) or not record["break_instruction"].strip():
            raise ValueError(f"record {index}: break_instruction must be a nonempty string")
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError(f"record {index}: split must be train, val, or test")
        splits.add(record["split"])
        if "source_state_id" in record:
            source_key = (record["case_id"], str(record["source_state_id"]))
            if source_key in source_splits and source_splits[source_key] != record["split"]:
                raise ValueError("the same source state cannot occur in multiple splits")
            source_splits[source_key] = record["split"]
        if "record_id" in record:
            identity = str(record["record_id"])
            if identity in seen:
                raise ValueError(f"duplicate record_id: {identity}")
            seen.add(identity)
        for key in ACTION_KEYS:
            vector = record[key]
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"record {index}: {key} must be a nonempty 1D list")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                raise ValueError(f"record {index}: {key} must contain finite numbers")
            dimensions.add(len(vector))
    if len(dimensions) != 1:
        raise ValueError("all action/reference vectors must share one dimension")
    if require_splits and not {"train", "val"} <= splits:
        raise ValueError("records must contain train and val splits")
    return dimensions.pop()


def export_triplet_records(source: str | Path, destination: str | Path, *, jsonl: bool = False) -> dict[str, Any]:
    """Package precomputed matched triplets, without model calls or inferred labels."""
    source = Path(source)
    if jsonl:
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    else:
        records = load_cached_records(source)
    dimension = validate_cached_records(records)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return {"num_records": len(records), "action_dim": dimension, "output": str(destination)}
