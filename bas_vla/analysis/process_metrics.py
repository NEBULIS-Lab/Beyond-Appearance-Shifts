from __future__ import annotations

from typing import Any


def build_process_metric_rows(
    reference_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    compare_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_map = {
        (int(ep["task_id"]), int(ep["episode_idx"])): ep
        for ep in reference_summary.get("episodes", [])
    }
    variant_map = {
        (int(ep["task_id"]), int(ep["episode_idx"])): ep
        for ep in variant_summary.get("episodes", [])
    }

    rows: list[dict[str, Any]] = []
    cumulative_reference = 0
    cumulative_variant = 0
    for order_idx, row in enumerate(compare_rows, start=1):
        key = (int(row["task_id"]), int(row["episode_idx"]))
        reference_ep = reference_map.get(key, {})
        variant_ep = variant_map.get(key, {})
        cumulative_reference += int(row["clean_success"])
        cumulative_variant += int(row["shift_success"])
        rows.append(
            {
                "order_idx": order_idx,
                "task_id": int(row["task_id"]),
                "episode_idx": int(row["episode_idx"]),
                "task_name": row["task_name"],
                "reference_success": int(row["clean_success"]),
                "variant_success": int(row["shift_success"]),
                "cumulative_reference_success_rate": cumulative_reference / order_idx,
                "cumulative_variant_success_rate": cumulative_variant / order_idx,
                "reference_steps": int(row["clean_executed_steps"]),
                "variant_steps": int(row["shift_executed_steps"]),
                "step_gap": float(row["step_gap"]),
                "mean_action_l2": float(row["mean_action_l2"]),
                "max_action_l2": float(row["max_action_l2"]),
                "mean_gripper_abs_diff": float(row["mean_gripper_abs_diff"]),
                "reference_replan_calls": int(reference_ep.get("replan_calls", 0)),
                "variant_replan_calls": int(variant_ep.get("replan_calls", 0)),
                "reference_action_trace_len": len(reference_ep.get("action_trace", [])),
                "variant_action_trace_len": len(variant_ep.get("action_trace", [])),
            }
        )
    return rows
