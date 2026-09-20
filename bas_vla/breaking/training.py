from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .residual_adapter import (
    ResidualAdapter,
    build_metric_row,
    build_vocab_from_texts,
    encode_instruction_bow,
    save_adapter_metadata,
)


@dataclass(frozen=True)
class TrainingConfig:
    objective_preset: str = "legacy"
    squared_error_reduction: str = "mean"
    lambda_imitation: float = 1.0
    lambda_expert_margin: float = 1.0
    hidden_dim: int = 64
    delta_scale: float = 0.35
    margin: float = 0.15
    lambda_consistency: float = 0.5
    lambda_margin: float = 1.0
    lambda_anchor: float = 0.2
    lr: float = 2e-3
    epochs: int = 60
    batch_size: int = 64


OBJECTIVE_PRESETS = ("legacy", "paper_three_component", "paper_without_consistency", "paper_without_separation")


def training_config(preset: str = "legacy", **overrides: Any) -> TrainingConfig:
    """Explicit executable objectives; presets do not identify evaluated checkpoints."""
    if preset not in OBJECTIVE_PRESETS:
        raise ValueError(f"unknown objective preset: {preset}")
    cfg = TrainingConfig(objective_preset=preset)
    if preset != "legacy":
        cfg = replace(cfg, squared_error_reduction="sum_last_mean", lambda_expert_margin=0.0, lambda_anchor=0.0)
    if preset == "paper_without_consistency":
        cfg = replace(cfg, lambda_consistency=0.0)
    if preset == "paper_without_separation":
        cfg = replace(cfg, lambda_margin=0.0)
    cfg = replace(cfg, **overrides)
    if cfg.squared_error_reduction not in {"mean", "sum_last_mean"}:
        raise ValueError("unsupported squared-error reduction")
    for key in ("lambda_imitation", "lambda_consistency", "lambda_margin", "lambda_expert_margin", "lambda_anchor"):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.hidden_dim < 1:
        raise ValueError("epochs, batch_size and hidden_dim must be positive")
    for key in ("lr", "margin", "delta_scale"):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    return cfg


def l2_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a - b, dim=-1)


def build_vocab_from_records(records: list[dict[str, Any]]) -> dict[str, int]:
    texts: list[str] = []
    for record in records:
        texts.extend(
            [
                str(record["clean_instruction"]),
                str(record["control_instruction"]),
                str(record["break_instruction"]),
            ]
        )
    return build_vocab_from_texts(texts)


def attach_bow_features(records: list[dict[str, Any]], vocab: dict[str, int]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["clean_bow"] = encode_instruction_bow(str(record["clean_instruction"]), vocab).tolist()
        item["control_bow"] = encode_instruction_bow(str(record["control_instruction"]), vocab).tolist()
        item["break_bow"] = encode_instruction_bow(str(record["break_instruction"]), vocab).tolist()
        enriched.append(item)
    return enriched


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        split = str(record.get("split", "train"))
        by_split[split].append(record)
    return by_split


def build_tensor_dataset(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    def stack(key: str) -> torch.Tensor:
        return torch.tensor(np.asarray([record[key] for record in records], dtype=np.float32), dtype=torch.float32)

    return {
        "clean_base": stack("clean_base_action"),
        "control_base": stack("control_base_action"),
        "break_base": stack("break_base_action"),
        "expert": stack("expert_action"),
        "clean_feat": stack("clean_bow"),
        "control_feat": stack("control_bow"),
        "break_feat": stack("break_bow"),
    }


def compute_losses(
    model: ResidualAdapter,
    batch: dict[str, torch.Tensor],
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    clean_hat = model(batch["clean_base"], batch["clean_feat"])
    control_hat = model(batch["control_base"], batch["control_feat"])
    break_hat = model(batch["break_base"], batch["break_feat"])

    def squared_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        errors = (a - b).square()
        if cfg.squared_error_reduction == "sum_last_mean":
            return errors.sum(dim=-1).mean()
        if cfg.squared_error_reduction != "mean":
            raise ValueError("unsupported squared-error reduction")
        return errors.mean()

    clean_mse = squared_error(clean_hat, batch["expert"])
    control_mse = squared_error(control_hat, batch["expert"])
    consistency = squared_error(control_hat, clean_hat)

    d_break_clean = l2_rows(break_hat, clean_hat)
    d_control_clean = l2_rows(control_hat, clean_hat)
    d_break_expert = l2_rows(break_hat, batch["expert"])
    d_control_expert = l2_rows(control_hat, batch["expert"])

    margin_clean = torch.relu(cfg.margin - d_break_clean + d_control_clean).mean()
    margin_expert = torch.relu(cfg.margin - d_break_expert + d_control_expert).mean()

    anchor = (
        F.mse_loss(clean_hat, batch["clean_base"])
        + F.mse_loss(control_hat, batch["control_base"])
        + 0.25 * F.mse_loss(break_hat, batch["break_base"])
    )

    total = (
        cfg.lambda_imitation * (clean_mse + control_mse)
        + cfg.lambda_consistency * consistency
        + cfg.lambda_margin * margin_clean
        + cfg.lambda_margin * cfg.lambda_expert_margin * margin_expert
        + cfg.lambda_anchor * anchor
    )

    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_clean_mse": float(clean_mse.detach().cpu()),
        "loss_control_mse": float(control_mse.detach().cpu()),
        "loss_consistency": float(consistency.detach().cpu()),
        "loss_margin_clean": float(margin_clean.detach().cpu()),
        "loss_margin_expert": float(margin_expert.detach().cpu()),
        "loss_anchor": float(anchor.detach().cpu()),
    }
    outputs = {"clean": clean_hat, "control": control_hat, "break": break_hat}
    return total, metrics, outputs


def summarize_case_metrics(per_case: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
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

    by_case = {case_id: summarize_rows(rows) for case_id, rows in per_case.items()}
    all_rows = [row for rows in per_case.values() for row in rows]
    summary = summarize_rows(all_rows)
    summary["by_case"] = by_case
    summary["num_records"] = len(all_rows)
    return summary


def evaluate_model(
    model: ResidualAdapter,
    dataset: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        batch = {key: value.to(device) for key, value in dataset.items()}
        clean = model(batch["clean_base"], batch["clean_feat"]).detach().cpu().numpy()
        control = model(batch["control_base"], batch["control_feat"]).detach().cpu().numpy()
        brk = model(batch["break_base"], batch["break_feat"]).detach().cpu().numpy()
        expert = dataset["expert"].cpu().numpy()

    per_case: dict[str, list[dict[str, float]]] = defaultdict(list)
    for idx, record in enumerate(records):
        per_case[str(record["case_id"])].append(
            build_metric_row(clean=clean[idx], control=control[idx], brk=brk[idx], expert=expert[idx])
        )
    return summarize_case_metrics(per_case)


def train_adapter(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    cfg: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    if not train_records:
        raise ValueError("train_records must not be empty")
    if not val_records:
        raise ValueError("val_records must not be empty")

    train_data = build_tensor_dataset(train_records)
    val_data = build_tensor_dataset(val_records)
    input_dim = int(train_data["clean_base"].shape[-1] + train_data["clean_feat"].shape[-1])
    output_dim = int(train_data["clean_base"].shape[-1])

    model = ResidualAdapter(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=output_dim,
        delta_scale=cfg.delta_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    train_size = train_data["expert"].shape[0]

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        permutation = torch.randperm(train_size)
        batch_metrics: list[dict[str, float]] = []
        for start in range(0, train_size, cfg.batch_size):
            indices = permutation[start : start + cfg.batch_size]
            batch = {key: value[indices].to(device) for key, value in train_data.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _ = compute_losses(model, batch, cfg)
            loss.backward()
            optimizer.step()
            batch_metrics.append(metrics)

        train_loss = float(np.mean([item["loss_total"] for item in batch_metrics]))
        model.eval()
        with torch.no_grad():
            val_batch = {key: value.to(device) for key, value in val_data.items()}
            val_loss, val_loss_metrics, _ = compute_losses(model, val_batch, cfg)
            val_summary = evaluate_model(model, val_data, val_records, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss_total": train_loss,
                "val_loss_total": float(val_loss.detach().cpu()),
                "val_mean_clean_ref_gap": val_summary.get("mean_clean_ref_gap"),
                "val_break_gt_control_rate_clean_ref": val_summary.get("break_gt_control_rate_clean_ref"),
                "val_mean_clean_mse": val_summary.get("mean_clean_mse"),
                "val_mean_control_mse": val_summary.get("mean_control_mse"),
                **{f"val_{key}": value for key, value in val_loss_metrics.items()},
            }
        )

        current_val = float(val_loss.detach().cpu())
        if current_val < best_val:
            best_val = current_val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training did not produce a best state")

    model.load_state_dict(best_state)
    train_summary = evaluate_model(model, train_data, train_records, device)
    val_summary = evaluate_model(model, val_data, val_records, device)
    return {
        "history": history,
        "best_val_loss_total": best_val,
        "train_metrics": train_summary,
        "val_metrics": val_summary,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "input_dim": input_dim,
        "output_dim": output_dim,
    }


def save_training_artifacts(
    output_dir: Path,
    result: dict[str, Any],
    vocab: dict[str, int],
    cfg: TrainingConfig,
    *,
    provenance: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "residual_adapter.pt"
    metadata_path = output_dir / "adapter_metadata.json"
    vocab_path = output_dir / "vocab.json"
    summary_path = output_dir / "results.json"

    torch.save(result["state_dict"], checkpoint_path)
    save_adapter_metadata(
        metadata_path,
        adapter_name="bas_vla_residual_adapter",
        input_dim=int(result["input_dim"]),
        hidden_dim=int(cfg.hidden_dim),
        output_dim=int(result["output_dim"]),
        delta_scale=float(cfg.delta_scale),
        vocab_size=len(vocab),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"feature_type": "bow", "feature_normalization": "binary_presence",
                     "objective_config": asdict(cfg), "provenance": provenance or {}})
    metadata.update(provenance or {})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2)

    serializable = {
        "config": asdict(cfg),
        "provenance": provenance or {},
        "best_val_loss_total": result["best_val_loss_total"],
        "train_metrics": result["train_metrics"],
        "val_metrics": result["val_metrics"],
        "history": result["history"],
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "metadata": metadata_path.name,
            "vocab": vocab_path.name,
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)
