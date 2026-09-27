from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SemanticBreakPair:
    pair_id: str
    family: str
    suite: str
    task_id: int
    clean_instruction: str
    control_instruction: str
    break_instruction: str
    changed_semantics: str
    notes: str = ""


def load_semantic_break_pairs(path: str | Path) -> list[SemanticBreakPair]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    pairs = payload.get("pairs", [])
    return [
        SemanticBreakPair(
            pair_id=str(item["pair_id"]),
            family=str(item["family"]),
            suite=str(item["suite"]),
            task_id=int(item["task_id"]),
            clean_instruction=str(item["clean_instruction"]),
            control_instruction=str(item["control_instruction"]),
            break_instruction=str(item["break_instruction"]),
            changed_semantics=str(item["changed_semantics"]),
            notes=str(item.get("notes", "")),
        )
        for item in pairs
    ]


def summarize_pairs(pairs: list[SemanticBreakPair]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_suite: dict[str, int] = {}
    task_keys: set[tuple[str, int]] = set()
    pair_ids: set[str] = set()
    duplicate_pair_ids: list[str] = []

    for pair in pairs:
        by_family[pair.family] = by_family.get(pair.family, 0) + 1
        by_suite[pair.suite] = by_suite.get(pair.suite, 0) + 1
        task_keys.add((pair.suite, pair.task_id))
        if pair.pair_id in pair_ids:
            duplicate_pair_ids.append(pair.pair_id)
        else:
            pair_ids.add(pair.pair_id)

    return {
        "num_pairs": len(pairs),
        "num_families": len(by_family),
        "num_suite_task_keys": len(task_keys),
        "families": dict(sorted(by_family.items())),
        "suites": dict(sorted(by_suite.items())),
        "duplicate_pair_ids": sorted(set(duplicate_pair_ids)),
    }
