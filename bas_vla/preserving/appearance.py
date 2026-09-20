"""Appearance-shift presets for evaluation and reproducing public benchmark slices.

This module defines perturbation presets used by public evaluation scripts. It
is not the preserving auxiliary itself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


APPEARANCE_SHIFT_PRESETS: dict[str, dict[str, Any]] = {
    "clean": {
        "name": "clean",
        "scene_properties": None,
        "image_shift": {},
    },
    "bg_dark_dim": {
        "name": "bg_dark_dim",
        "scene_properties": {
            "floor_style": "dark",
            "wall_style": "dark-blue",
        },
        "image_shift": {
            "brightness": 0.64,
            "contrast": 0.90,
            "gamma": 1.22,
        },
    },
    "warm_color_cast": {
        "name": "warm_color_cast",
        "scene_properties": None,
        "image_shift": {
            "brightness": 0.95,
            "contrast": 1.05,
            "channel_scale": [1.10, 0.98, 0.86],
        },
    },
    "style_swap": {
        "name": "style_swap",
        "scene_properties": {
            "floor_style": "rustic",
            "wall_style": "dark-blue",
        },
        "image_shift": {},
    },
    "margin_noise_band": {
        "name": "margin_noise_band",
        "scene_properties": None,
        "image_shift": {
            "kind": "margin_noise_band",
            "top_fraction": 0.16,
            "side_fraction": 0.10,
            "noise_std": 40.0,
        },
    },
}


def get_appearance_shift_spec(name: str) -> dict[str, Any]:
    if name not in APPEARANCE_SHIFT_PRESETS:
        raise KeyError(f"unknown appearance shift preset: {name}")
    return deepcopy(APPEARANCE_SHIFT_PRESETS[name])


def get_scene_properties(name: str) -> dict[str, Any] | None:
    return deepcopy(get_appearance_shift_spec(name).get("scene_properties"))


def get_image_shift(name: str) -> dict[str, Any]:
    return deepcopy(get_appearance_shift_spec(name).get("image_shift", {}))
