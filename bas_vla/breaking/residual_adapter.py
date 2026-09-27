from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab_from_texts(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_instruction_bow(text: str, vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for token in tokenize(text):
        token_idx = vocab.get(token)
        if token_idx is not None:
            vec[token_idx] = 1.0
    return vec


class ResidualAdapter(nn.Module):
    """Small text-conditioned residual adapter used in the current breaking branch."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, delta_scale: float) -> None:
        super().__init__()
        self.delta_scale = delta_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, action: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([action, features], dim=-1))
        return action + self.delta_scale * torch.tanh(delta)


def save_adapter_metadata(
    path: Path,
    *,
    adapter_name: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    delta_scale: float,
    vocab_size: int,
) -> None:
    payload = {
        "adapter_name": adapter_name,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "delta_scale": delta_scale,
        "vocab_size": vocab_size,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_adapter_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_residual_adapter(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    device: torch.device,
) -> ResidualAdapter:
    metadata = load_adapter_metadata(metadata_path)
    adapter = ResidualAdapter(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        output_dim=int(metadata["output_dim"]),
        delta_scale=float(metadata["delta_scale"]),
    )
    state_dict = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    adapter.load_state_dict(state_dict)
    adapter = adapter.to(device)
    adapter.eval()
    return adapter


def apply_residual_adapter(
    adapter: ResidualAdapter,
    actions: np.ndarray,
    feature_vec: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    actions_np = np.asarray(actions, dtype=np.float32)
    if actions_np.ndim == 1:
        actions_np = actions_np[None, :]
    feature_np = np.asarray(feature_vec, dtype=np.float32)
    feature_batch = np.repeat(feature_np[None, :], actions_np.shape[0], axis=0)
    with torch.no_grad():
        action_tensor = torch.tensor(actions_np, dtype=torch.float32, device=device)
        feature_tensor = torch.tensor(feature_batch, dtype=torch.float32, device=device)
        calibrated = adapter(action_tensor, feature_tensor).detach().cpu().numpy()
    return np.asarray(calibrated, dtype=np.float32)


def build_metric_row(
    clean: np.ndarray,
    control: np.ndarray,
    brk: np.ndarray,
    expert: np.ndarray,
) -> dict[str, float]:
    return {
        "clean_mse": float(np.mean((clean - expert) ** 2)),
        "control_mse": float(np.mean((control - expert) ** 2)),
        "break_clean_l2": float(np.linalg.norm(brk - clean)),
        "control_clean_l2": float(np.linalg.norm(control - clean)),
        "break_expert_l2": float(np.linalg.norm(brk - expert)),
        "control_expert_l2": float(np.linalg.norm(control - expert)),
    }


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}

    keys = sorted(rows[0].keys())
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        summary[f"mean_{key}"] = float(values.mean())

    break_clean = np.asarray([row["break_clean_l2"] for row in rows], dtype=np.float32)
    control_clean = np.asarray([row["control_clean_l2"] for row in rows], dtype=np.float32)
    break_expert = np.asarray([row["break_expert_l2"] for row in rows], dtype=np.float32)
    control_expert = np.asarray([row["control_expert_l2"] for row in rows], dtype=np.float32)

    summary["break_gt_control_rate_clean_ref"] = float((break_clean > control_clean).mean())
    summary["break_gt_control_rate_expert_ref"] = float((break_expert > control_expert).mean())
    summary["mean_clean_ref_gap"] = float((break_clean - control_clean).mean())
    summary["mean_expert_ref_gap"] = float((break_expert - control_expert).mean())
    return summary


def summarize_record_set(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_case: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        clean = np.asarray(record["clean"], dtype=np.float32)
        control = np.asarray(record["control"], dtype=np.float32)
        brk = np.asarray(record["break"], dtype=np.float32)
        expert = np.asarray(record["expert"], dtype=np.float32)
        per_case[str(record["case_id"])].append(
            build_metric_row(clean=clean, control=control, brk=brk, expert=expert)
        )

    by_case = {case_id: _summarize_rows(rows) for case_id, rows in per_case.items()}
    all_rows = [row for rows in per_case.values() for row in rows]
    summary = _summarize_rows(all_rows)
    summary["by_case"] = by_case
    summary["num_records"] = len(all_rows)
    return summary
