#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.preserving.appearance import APPEARANCE_SHIFT_PRESETS, get_appearance_shift_spec
from bas_vla.runtime.calibration import CalibrationRuntime, add_calibration_arguments
from bas_vla.runtime import (
    build_openpi_observation,
    env_path,
    require_path,
)
from bas_vla.integrations.openpi_libero import (
    apply_coverage_compat,
    ensure_runtime_env,
    get_dummy_action,
    get_max_steps,
    preprocess_image,
    quat_to_axis_angle,
    register_external_roots,
    save_contact_sheet,
    save_rollout_gif,
    select_task_ids,
    time_stamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local OpenPI checkpoint evaluation on LIBERO with appearance shifts."
    )
    parser.add_argument("--openpi-root", type=Path, default=env_path("BAS_OPENPI_ROOT"))
    parser.add_argument("--libero-root", type=Path, default=env_path("BAS_LIBERO_ROOT"))
    parser.add_argument("--libero-site-packages", type=Path, default=env_path("BAS_LIBERO_SITE_PACKAGES"))
    parser.add_argument("--libero-config-path", type=Path, default=env_path("LIBERO_CONFIG_PATH"))
    parser.add_argument("--checkpoint-dir", type=Path, default=env_path("BAS_OPENPI_CHECKPOINT"))
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--episode-indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shift-preset", default="clean", choices=sorted(APPEARANCE_SHIFT_PRESETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction-override", default=None)
    parser.add_argument("--run-note", default="")
    parser.add_argument("--save-rollout-frames-limit", type=int, default=0)
    parser.add_argument("--save-rollout-frame-stride", type=int, default=1)
    parser.add_argument("--save-rollout-media", action="store_true")
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    add_calibration_arguments(parser)
    return parser.parse_args()


def make_libero_env(task: Any, resolution: int, seed: int, shift_spec: dict[str, Any]) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_kwargs: dict[str, Any] = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if shift_spec.get("scene_properties") is not None:
        env_kwargs["scene_properties"] = shift_spec["scene_properties"]
    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(seed)
    return env, task_description


def select_episode_indices(initial_states: Any, requested_indices: list[int] | None, num_trials_per_task: int) -> list[int]:
    total = len(initial_states)
    if requested_indices is not None:
        if not requested_indices or len(set(requested_indices)) != len(requested_indices) or any(idx < 0 or idx >= total for idx in requested_indices):
            raise ValueError("episode indices must be nonempty, unique, and in range")
        return list(requested_indices)
    return list(range(min(num_trials_per_task, total)))


def build_policy_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )


def build_policy_element(
    full_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    instruction: str,
) -> dict[str, Any]:
    return {
        "observation/image": full_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": str(instruction),
    }


def infer_action_chunk(policy: Any, element: dict[str, Any], replan_steps: int) -> tuple[np.ndarray, dict[str, float]]:
    infer_out = policy.infer(element)
    action_chunk = np.asarray(infer_out["actions"], dtype=np.float32)
    timing = {
        key: float(value)
        for key, value in infer_out.get("policy_timing", {}).items()
        if isinstance(value, (int, float))
    }
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]
    if action_chunk.shape[0] < replan_steps:
        raise RuntimeError(
            f"policy returned {action_chunk.shape[0]} actions, smaller than replan_steps={replan_steps}"
        )
    return action_chunk[:replan_steps], timing


def main() -> int:
    args = parse_args()
    openpi_root = require_path(args.openpi_root, "--openpi-root", "BAS_OPENPI_ROOT")
    libero_root = require_path(args.libero_root, "--libero-root", "BAS_LIBERO_ROOT")
    checkpoint_dir = require_path(args.checkpoint_dir, "--checkpoint-dir", "BAS_OPENPI_CHECKPOINT")

    apply_coverage_compat()
    register_external_roots(
        openpi_root=openpi_root,
        libero_root=libero_root,
        libero_site_packages=args.libero_site_packages,
    )
    ensure_runtime_env(args.libero_config_path)

    from libero.libero import benchmark
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    torch_load = torch.load

    def torch_load_compat(*load_args: Any, **load_kwargs: Any) -> Any:
        load_kwargs.setdefault("weights_only", False)
        return torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_compat

    shift_spec = get_appearance_shift_spec(args.shift_preset)
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_ids = select_task_ids(task_suite, args.task_ids, args.max_tasks)
    max_steps = args.max_steps_override or get_max_steps(args.suite)

    output_dir = args.output_dir / (
        f"openpi_pi05_libero__{args.suite}__task"
        f"{'-'.join(str(task_id) for task_id in task_ids)}__{args.shift_preset}__{time_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = openpi_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    calibration = CalibrationRuntime.from_args(args, policy)

    def infer_carrier(observation, instruction):
        return infer_action_chunk(policy, build_policy_element(observation["full_image"],
            observation["wrist_image"], observation["state"], instruction), args.replan_steps)

    all_episode_records: list[dict[str, Any]] = []
    total_successes = 0
    total_episodes = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = make_libero_env(task, args.resolution, args.seed, shift_spec)
        episode_indices = select_episode_indices(initial_states, args.episode_indices, args.num_trials_per_task)
        for episode_idx in episode_indices:
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            action_plan: deque[np.ndarray] = deque()
            action_trace: list[list[float]] = []
            preserving_trace: list[dict[str, Any]] = []
            rollout_frames: list[np.ndarray] = []
            instruction_used = args.instruction_override or task_description
            success = False
            t = 0
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(get_dummy_action())
                    t += 1
                    continue

                base_img, wrist_img = preprocess_image(obs, args.resize_size, shift_spec, noise_seed=t)
                if args.save_rollout_frames_limit > 0:
                    if len(rollout_frames) < args.save_rollout_frames_limit and (
                        (t - args.num_steps_wait) % max(1, args.save_rollout_frame_stride) == 0
                    ):
                        rollout_frames.append(base_img)
                if not action_plan:
                    policy_state = build_policy_state(obs)
                    carrier_observation = build_openpi_observation(agentview_image=base_img,
                        wrist_image=wrist_img, state=policy_state)
                    action_chunk, chunk_trace = calibration.predict(carrier_observation, instruction_used,
                        infer_carrier, step_index=max(0, t - args.num_steps_wait), task_id=task_id)
                    preserving_trace.append(chunk_trace)

                    action_plan.extend(action_chunk[: args.replan_steps])

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                action_trace.append(action.tolist())
                obs, _, done, _ = env.step(action.tolist())
                success = bool(done)
                t += 1
                if done:
                    break

            media_paths: dict[str, str] = {}
            if args.save_rollout_media and rollout_frames:
                media_dir = output_dir / "media"
                gif_path = media_dir / f"task{task_id:02d}_ep{episode_idx:02d}.gif"
                contact_path = media_dir / f"task{task_id:02d}_ep{episode_idx:02d}_contact.png"
                save_rollout_gif(rollout_frames, gif_path, args.gif_duration_ms)
                save_contact_sheet(rollout_frames, contact_path)
                media_paths = {
                    "gif": str(gif_path),
                    "contact_sheet": str(contact_path),
                }

            all_episode_records.append(
                {
                    "model_alias": "openpi_pi05_libero",
                    "suite": args.suite,
                    "task_id": task_id,
                    "task_description": task_description,
                    "instruction_used": instruction_used,
                    "shift_preset": args.shift_preset,
                    "shift_spec": shift_spec,
                    "episode_idx": episode_idx,
                    "success": success,
                    "steps": t,
                    "checkpoint_path": str(checkpoint_dir),
                    "action_trace": action_trace,
                    "preserving_mode": calibration.mode,
                    "preserving_trace": preserving_trace,
                    "rollout_media": media_paths,
                }
            )
            total_episodes += 1
            total_successes += int(success)
        env.close()

    summary = {
        "model_alias": "openpi_pi05_libero",
        "suite": args.suite,
        "checkpoint_path": str(checkpoint_dir),
        "shift_preset": args.shift_preset,
        "shift_spec": shift_spec,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "num_open_loop_steps": args.replan_steps,
        "seed": args.seed,
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", "cuda"),
        "run_note": args.run_note,
        "instruction_override": args.instruction_override,
        "preserving_mode": calibration.mode,
        "preserving_phase_horizon_steps": args.preserving_phase_horizon_steps,
        "per_episode": all_episode_records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[bas-vla] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
