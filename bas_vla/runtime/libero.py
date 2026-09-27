from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass(frozen=True)
class FormalTaskTriple:
    gpu: str
    task_id: int
    clean_instruction: str
    control_instruction: str
    break_instruction: str
    pair_prefix: str
    seeds: str


@dataclass(frozen=True)
class AppearanceFormalRow:
    gpu: str
    task_id: int
    instruction: str
    shift_preset: str
    pair_prefix: str
    seeds: str


def get_max_steps(suite_name: str) -> int:
    if suite_name not in SUITE_MAX_STEPS:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    return SUITE_MAX_STEPS[suite_name]


def select_task_ids(task_suite: Any, requested_task_ids: list[int] | None, max_tasks: int | None) -> list[int]:
    if requested_task_ids:
        return requested_task_ids
    all_task_ids = list(range(task_suite.n_tasks))
    if max_tasks is not None:
        return all_task_ids[:max_tasks]
    return all_task_ids


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def load_formal_task_triples(path: Path) -> list[FormalTaskTriple]:
    return [
        FormalTaskTriple(
            gpu=item["gpu"],
            task_id=int(item["task_id"]),
            clean_instruction=item["clean_instruction"],
            control_instruction=item["control_instruction"],
            break_instruction=item["break_instruction"],
            pair_prefix=item["pair_prefix"],
            seeds=item["seeds"],
        )
        for item in _read_tsv(path)
    ]


def load_appearance_formal_rows(path: Path) -> list[AppearanceFormalRow]:
    return [
        AppearanceFormalRow(
            gpu=item["gpu"],
            task_id=int(item["task_id"]),
            instruction=item["instruction"],
            shift_preset=item["shift_preset"],
            pair_prefix=item["pair_prefix"],
            seeds=item["seeds"],
        )
        for item in _read_tsv(path)
    ]
