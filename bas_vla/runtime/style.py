"""Geometry-preserving LIBERO scene style selection for OFT evaluation."""
from __future__ import annotations
from pathlib import Path


def scene_style_properties(mode, floor_style="rustic", wall_style="dark-blue"):
    if mode == "clean":
        return None
    if mode != "style_swap":
        raise ValueError(f"unknown scene style: {mode}")
    return {"floor_style": floor_style, "wall_style": wall_style}


def make_oft_style_env(task, *, resolution, seed, mode, floor_style, wall_style):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    kwargs = dict(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                  camera_heights=resolution, camera_widths=resolution)
    properties = scene_style_properties(mode, floor_style, wall_style)
    if properties is not None:
        kwargs["scene_properties"] = properties
    env = OffScreenRenderEnv(**kwargs)
    env.seed(seed)
    return env
