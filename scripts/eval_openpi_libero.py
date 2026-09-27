#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.integrations.openpi_libero import (
    IMAGE_SHIFT_PRESETS,
    apply_coverage_compat,
    ensure_runtime_env,
    get_dummy_action,
    get_max_steps,
    load_distractor_assets,
    preprocess_image,
    quat_to_axis_angle,
    register_external_roots,
    save_contact_sheet,
    save_rollout_gif,
    select_task_ids,
    time_stamp,
)
from bas_vla.runtime.calibration import CalibrationRuntime, add_calibration_arguments
from bas_vla.runtime import env_path, require_path, build_openpi_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local OpenPI checkpoint evaluation on LIBERO.")
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
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction-override", default=None)
    parser.add_argument("--instruction-tag", default="")
    parser.add_argument("--pair-id", default="")
    parser.add_argument("--run-note", default="")
    parser.add_argument("--save-action-trace-limit", type=int, default=0)
    parser.add_argument("--save-rollout-frames-limit", type=int, default=0)
    parser.add_argument("--save-rollout-frame-stride", type=int, default=1)
    parser.add_argument("--save-rollout-media", action="store_true")
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    parser.add_argument("--image-shift-preset", default="clean", choices=sorted(IMAGE_SHIFT_PRESETS))
    parser.add_argument("--distractor-manifest", type=Path, default=None)
    add_calibration_arguments(parser)
    return parser.parse_args()


def make_libero_env(task: Any, resolution: int, seed: int) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


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

    np.random.seed(args.seed)
    image_shift_cfg = dict(IMAGE_SHIFT_PRESETS[args.image_shift_preset])
    distractor_assets = load_distractor_assets(args.distractor_manifest)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_ids = select_task_ids(task_suite, args.task_ids, args.max_tasks)
    max_steps = args.max_steps_override or get_max_steps(args.suite)

    output_dir = args.output_dir / (
        f"openpi_pi05_libero__{args.suite}__"
        f"{args.pair_id or args.instruction_tag or 'run'}__"
        f"{args.instruction_tag or 'clean'}__{time_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = openpi_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)

    calibration = CalibrationRuntime.from_args(args, policy)

    def infer_carrier(observation, instruction):
        result = policy.infer({"observation/image": observation["full_image"],
            "observation/wrist_image": observation["wrist_image"],
            "observation/state": observation["state"], "prompt": instruction})
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if len(actions) < args.replan_steps:
            raise ValueError("carrier chunk shorter than replan_steps")
        return actions[:args.replan_steps], result.get("policy_timing", {})

    all_episode_records: list[dict[str, Any]] = []
    total_successes = 0
    total_episodes = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = make_libero_env(task, args.resolution, args.seed)
        for episode_idx in range(min(args.num_trials_per_task, len(initial_states))):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            action_plan: deque[np.ndarray] = deque()
            action_trace: list[list[float]] = []
            rollout_frames: list[np.ndarray] = []
            chunk_trace: list[dict[str, Any]] = []
            policy_timing_trace: list[dict[str, float]] = []
            instruction_used = args.instruction_override or task_description
            success = False
            t = 0
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(get_dummy_action())
                    t += 1
                    continue

                step_seed = args.seed * 1_000_000 + task_id * 10_000 + episode_idx * 100 + t
                base_img, wrist_img = preprocess_image(
                    obs,
                    args.resize_size,
                    image_shift_cfg,
                    noise_seed=step_seed,
                    distractor_assets=distractor_assets,
                )
                if args.save_rollout_frames_limit > 0:
                    if len(rollout_frames) < args.save_rollout_frames_limit and (
                        (t - args.num_steps_wait) % max(1, args.save_rollout_frame_stride) == 0
                    ):
                        rollout_frames.append(base_img)

                if not action_plan:
                    carrier_observation = build_openpi_observation(agentview_image=base_img,
                        wrist_image=wrist_img, state=np.concatenate((obs["robot0_eef_pos"],
                        quat_to_axis_angle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])))
                    action_chunk, trace = calibration.predict(carrier_observation, instruction_used,
                        infer_carrier, step_index=max(0, t - args.num_steps_wait), task_id=task_id)
                    chunk_trace.append(trace)
                    policy_timing_trace.append(trace["timing"])
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                if args.save_action_trace_limit <= 0 or len(action_trace) < args.save_action_trace_limit:
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
                    "instruction_tag": args.instruction_tag,
                    "pair_id": args.pair_id,
                    "episode_idx": episode_idx,
                    "success": success,
                    "steps": t,
                    "checkpoint_path": str(checkpoint_dir),
                    "action_trace": action_trace,
                    "policy_timing_trace": policy_timing_trace,
                    "mode": calibration.mode,
                    "chunk_trace": chunk_trace,
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
        "status": "completed",
        "mode": calibration.mode,
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
        "instruction_tag": args.instruction_tag,
        "pair_id": args.pair_id,
        "config_name": args.config_name,
        "image_shift_preset": args.image_shift_preset,
        "image_shift_spec": image_shift_cfg,
        "per_episode": all_episode_records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[bas-vla] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
