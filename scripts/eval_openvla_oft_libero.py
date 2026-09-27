#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import site
import sys
import tempfile
import types
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.pairs import load_semantic_break_pairs
from bas_vla.runtime.style import make_oft_style_env
from bas_vla.runtime.calibration import CalibrationRuntime, add_calibration_arguments
from bas_vla.runtime import (
    get_max_steps,
    resolve_openvla_oft_runtime,
)
from bas_vla.integrations.openpi_libero import save_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline OpenVLA-OFT checkpoint evaluation on LIBERO semantic-break pairs.")
    parser.add_argument("--pairs-config", type=Path, default=REPO_ROOT / "configs" / "openvla_oft" / "semantic_break_pairs_main.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--num-open-loop-steps", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--env-img-res", type=int, default=256)
    parser.add_argument("--save-rollout-media", action="store_true")
    parser.add_argument("--gif-duration-ms", type=int, default=160)
    parser.add_argument("--scene-style", choices=("clean", "style_swap"), default="clean")
    parser.add_argument("--floor-style", default="rustic")
    parser.add_argument("--wall-style", default="dark-blue")
    parser.add_argument("--episode-indices", nargs="+", type=int,
                        help="Explicit matched initial-state indices; identical for all instruction variants")
    add_calibration_arguments(parser)
    return parser.parse_args()


def ensure_libero_config(libero_root: Path, datasets_root: Path, target_root: Path) -> Path:
    config_root = target_root / "libero_config"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    package_root = libero_root / "src" / "LIBERO" / "libero" / "libero"
    content = "\n".join(
        [
            f"benchmark_root: {package_root}",
            f"bddl_files: {package_root / 'bddl_files'}",
            f"init_states: {package_root / 'init_files'}",
            f"datasets: {datasets_root}",
            f"assets: {package_root / 'assets'}",
            "",
        ]
    )
    config_path.write_text(content, encoding="utf-8")
    return config_root


def bootstrap_runtime(runtime_root: Path, runtime: Any) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if runtime.libero_config_path is not None:
        os.environ["LIBERO_CONFIG_PATH"] = str(runtime.libero_config_path)
    else:
        if runtime.libero_root is None or runtime.datasets_root is None:
            raise SystemExit(
                "OpenVLA-OFT evaluation requires LIBERO_CONFIG_PATH or both BAS_LIBERO_ROOT and BAS_LIBERO_DATASETS_ROOT."
            )
        generated = ensure_libero_config(runtime.libero_root, runtime.datasets_root, runtime_root)
        os.environ["LIBERO_CONFIG_PATH"] = str(generated)

    sys.path.insert(0, str(runtime.repo_root))
    if runtime.site_packages and runtime.site_packages.is_dir():
        site.addsitedir(str(runtime.site_packages))
    if runtime.lerobot_site_packages and runtime.lerobot_site_packages.is_dir():
        site.addsitedir(str(runtime.lerobot_site_packages))
    if runtime.libero_root:
        libero_src = runtime.libero_root / "src" / "LIBERO"
        if libero_src.is_dir():
            sys.path.insert(0, str(libero_src))
        sys.path.insert(0, str(runtime.libero_root))

    from prismatic.vla.constants import NormalizationType

    stub = types.ModuleType("prismatic.vla.datasets.rlds.utils.data_utils")
    stub.NormalizationType = NormalizationType
    sys.modules["prismatic.vla.datasets.rlds.utils.data_utils"] = stub

    import experiments.robot.openvla_utils as openvla_utils

    openvla_utils.update_auto_map = lambda *_args, **_kwargs: None
    openvla_utils.check_model_logic_mismatch = lambda *_args, **_kwargs: None


def flatten_actions(actions: Any) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def infer_openvla_action_chunk(
    *,
    cfg: Any,
    observation: dict[str, Any],
    instruction: str,
    model_bundle: dict[str, Any],
) -> np.ndarray:
    actions = model_bundle["get_action"](
        cfg,
        model_bundle["model"],
        observation,
        instruction,
        processor=model_bundle["processor"],
        action_head=model_bundle["action_head"],
        proprio_projector=model_bundle["proprio_projector"],
        noisy_action_projector=model_bundle["noisy_action_projector"],
        use_film=cfg.use_film,
    )
    chunk = flatten_actions(actions)
    if len(chunk) < cfg.num_open_loop_steps:
        raise ValueError("carrier chunk shorter than num_open_loop_steps")
    return chunk[: cfg.num_open_loop_steps]


def load_init_states_compat(task_suite: Any, task_id: int) -> Any:
    original_torch_load = torch.load

    def wrapped_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
        path = str(load_args[0]) if load_args else ""
        if path.endswith(".pruned_init") and "weights_only" not in load_kwargs:
            load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = wrapped_torch_load
    try:
        return task_suite.get_task_init_states(task_id)
    finally:
        torch.load = original_torch_load


def rotate_180(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    return arr[::-1, ::-1]


def save_rollout_gif(frames: list[np.ndarray], output_path: Path, duration_ms: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, duration=duration_ms / 1000.0) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def build_cfg(generate_config_cls: Any, pair: Any, args: argparse.Namespace, checkpoint_path: Path) -> Any:
    return generate_config_cls(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint_path),
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=bool(args.center_crop),
        num_open_loop_steps=args.num_open_loop_steps,
        lora_rank=32,
        unnorm_key=pair.suite,
        load_in_8bit=bool(args.load_in_8bit),
        load_in_4bit=bool(args.load_in_4bit),
        task_suite_name=pair.suite,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=args.num_trials,
        initial_states_path="DEFAULT",
        env_img_res=args.env_img_res,
        run_id_note=f"openvla-oft-{pair.pair_id}",
        local_log_dir=str(args.output_dir),
        use_wandb=False,
        seed=args.seed,
    )


def run_episode_collect(
    *,
    cfg: Any,
    env: Any,
    initial_state: np.ndarray,
    instruction: str,
    model_bundle: dict[str, Any],
    save_media: bool,
    calibration: CalibrationRuntime,
    task_id: int | None = None,
) -> dict[str, Any]:
    prepare_observation = model_bundle["prepare_observation"]
    process_action = model_bundle["process_action"]
    get_libero_dummy_action = model_bundle["get_libero_dummy_action"]

    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue: deque[np.ndarray] = deque(maxlen=cfg.num_open_loop_steps)
    action_trace: list[list[float]] = []
    preserving_trace: list[dict[str, Any]] = []
    captured_frames: list[np.ndarray] = []
    max_steps = int(model_bundle["task_max_steps"][cfg.task_suite_name])

    t = 0
    success = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            if done:
                success = True
                break
            continue

        observation, _ = prepare_observation(obs, model_bundle["resize_size"])
        if save_media:
            captured_frames.append(rotate_180(obs["agentview_image"]))

        if not action_queue:
            def infer_carrier(current_observation, current_instruction):
                return infer_openvla_action_chunk(cfg=cfg, observation=current_observation,
                    instruction=current_instruction, model_bundle=model_bundle)
            actions_arr, chunk_trace = calibration.predict(observation, instruction, infer_carrier,
                step_index=max(0, t - cfg.num_steps_wait), task_id=task_id)
            preserving_trace.append(chunk_trace)

            action_queue.extend(actions_arr[: cfg.num_open_loop_steps])

        action = np.asarray(action_queue.popleft(), dtype=np.float32)
        action_trace.append(action.tolist())
        action_env = process_action(action, cfg.model_family)
        obs, _, done, _ = env.step(action_env.tolist())
        t += 1
        if done:
            success = True
            break

    return {
        "success": bool(success),
        "steps": int(t),
        "action_trace": action_trace,
        "preserving_trace": preserving_trace,
        "frames": captured_frames,
    }


def main() -> int:
    args = parse_args()
    runtime = resolve_openvla_oft_runtime()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_runtime(output_dir / "runtime", runtime)

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
        TASK_MAX_STEPS,
        initialize_model,
        prepare_observation,
        process_action,
        validate_config,
    )
    from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere

    set_seed_everywhere(args.seed)
    pairs = load_semantic_break_pairs(args.pairs_config)
    if args.pair_id:
        selected = set(args.pair_id)
        pairs = [pair for pair in pairs if pair.pair_id in selected]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise SystemExit("no semantic-break pairs selected")

    benchmark_dict = benchmark.get_benchmark_dict()
    model_cache: dict[tuple[str, str], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for pair in pairs:
        cfg = build_cfg(GenerateConfig, pair, args, runtime.checkpoint_path)
        validate_config(cfg)

        cache_key = (str(runtime.checkpoint_path), pair.suite)
        if cache_key not in model_cache:
            model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
            resize_size = get_image_resize_size(cfg)
            model_cache[cache_key] = {
                "model": model,
                "action_head": action_head,
                "proprio_projector": proprio_projector,
                "noisy_action_projector": noisy_action_projector,
                "processor": processor,
                "resize_size": resize_size,
                "prepare_observation": prepare_observation,
                "process_action": process_action,
                "get_action": get_action,
                "get_libero_dummy_action": get_libero_dummy_action,
                "task_max_steps": TASK_MAX_STEPS,
            }

        if "calibration" not in model_cache[cache_key]:
            model_cache[cache_key]["calibration"] = CalibrationRuntime.from_args(
                args, model_cache[cache_key]["model"], action_space="oft_pre_process_action")
        calibration = model_cache[cache_key]["calibration"]
        task_suite = benchmark_dict[pair.suite]()
        task = task_suite.get_task(pair.task_id)
        initial_states = load_init_states_compat(task_suite, pair.task_id)
        env = make_oft_style_env(task, resolution=cfg.env_img_res, seed=args.seed,
            mode=args.scene_style, floor_style=args.floor_style, wall_style=args.wall_style)
        episode_indices = args.episode_indices if args.episode_indices is not None else list(range(min(args.num_trials, len(initial_states))))
        if len(set(episode_indices)) != len(episode_indices) or any(i < 0 or i >= len(initial_states) for i in episode_indices):
            env.close()
            raise ValueError("episode indices must be unique and in range")

        try:
            for episode_idx in episode_indices:
                initial_state = np.asarray(initial_states[episode_idx])
                clean_result = run_episode_collect(
                    cfg=cfg,
                    env=env,
                    initial_state=initial_state,
                    instruction=pair.clean_instruction,
                    model_bundle=model_cache[cache_key],
                    save_media=bool(args.save_rollout_media),
                    calibration=calibration,
                    task_id=pair.task_id,
                )
                control_result = run_episode_collect(
                    cfg=cfg,
                    env=env,
                    initial_state=initial_state,
                    instruction=pair.control_instruction,
                    model_bundle=model_cache[cache_key],
                    save_media=False,
                    calibration=calibration,
                    task_id=pair.task_id,
                )
                break_result = run_episode_collect(
                    cfg=cfg,
                    env=env,
                    initial_state=initial_state,
                    instruction=pair.break_instruction,
                    model_bundle=model_cache[cache_key],
                    save_media=bool(args.save_rollout_media),
                    calibration=calibration,
                    task_id=pair.task_id,
                )

                media = {}
                if args.save_rollout_media and clean_result["frames"] and break_result["frames"]:
                    media_root = output_dir / "media"
                    stem = f"{pair.pair_id}__trial{episode_idx:02d}"
                    clean_gif = media_root / "gifs" / f"{stem}__clean.gif"
                    break_gif = media_root / "gifs" / f"{stem}__break.gif"
                    contact = media_root / "contact_sheets" / f"{stem}.png"
                    save_rollout_gif(clean_result["frames"], clean_gif, args.gif_duration_ms)
                    save_rollout_gif(break_result["frames"], break_gif, args.gif_duration_ms)
                    save_contact_sheet(
                        [*clean_result["frames"][:5], *break_result["frames"][:5]],
                        contact,
                    )
                    media = {
                        "clean_gif": str(clean_gif),
                        "break_gif": str(break_gif),
                        "contact_sheet": str(contact),
                    }

                records.append(
                    {
                        "pair_id": pair.pair_id,
                        "family": pair.family,
                        "suite": pair.suite,
                        "task_id": pair.task_id,
                        "episode_idx": episode_idx,
                        "clean_success": clean_result["success"],
                        "control_success": control_result["success"],
                        "break_success": break_result["success"],
                        "clean_steps": clean_result["steps"],
                        "control_steps": control_result["steps"],
                        "break_steps": break_result["steps"],
                        "preserving_mode": calibration.mode,
                        "scene_style": args.scene_style,
                        "floor_style": args.floor_style,
                        "wall_style": args.wall_style,
                        "clean_preserving_trace": clean_result["preserving_trace"],
                        "control_preserving_trace": control_result["preserving_trace"],
                        "break_preserving_trace": break_result["preserving_trace"],
                        "media": media,
                    }
                )
        finally:
            env.close()

    summary = {
        "model_alias": "openvla_oft",
        "checkpoint_path": str(runtime.checkpoint_path),
        "pairs_config": str(args.pairs_config),
        "preserving_mode": calibration.mode,
        "scene_style": args.scene_style,
        "floor_style": args.floor_style,
        "wall_style": args.wall_style,
        "preserving_phase_horizon_steps": args.preserving_phase_horizon_steps,
        "num_records": len(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
