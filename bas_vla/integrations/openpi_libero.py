from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SHIFT_PRESETS: dict[str, dict[str, Any]] = {
    "clean": {},
    "margin_noise_band": {
        "kind": "margin_noise_band",
        "top_fraction": 0.16,
        "side_fraction": 0.10,
        "noise_std": 40.0,
    },
    "margin_noise_band_hard": {
        "kind": "margin_noise_band",
        "top_fraction": 0.20,
        "side_fraction": 0.12,
        "noise_std": 55.0,
    },
    "perimeter_clutter_noise_mix": {
        "kind": "perimeter_clutter_noise_mix",
        "top_fraction": 0.18,
        "side_fraction": 0.10,
        "noise_std": 30.0,
        "asset_target_height_fraction": 0.20,
        "asset_alpha": 225,
    },
}


def apply_coverage_compat() -> None:
    try:
        import coverage  # type: ignore

        if hasattr(coverage, "types"):
            if not hasattr(coverage.types, "Tracer"):

                class _CoverageTracerCompat:  # pragma: no cover
                    pass

                coverage.types.Tracer = _CoverageTracerCompat
            for name in (
                "TTraceData",
                "TShouldTraceFn",
                "TFileDisposition",
                "TShouldStartContextFn",
                "TWarnFn",
                "TTraceFn",
            ):
                if not hasattr(coverage.types, name):
                    setattr(coverage.types, name, object)
    except Exception:
        pass


def register_external_roots(
    *,
    openpi_root: Path,
    libero_root: Path,
    libero_site_packages: Path | None = None,
) -> None:
    candidates = [
        openpi_root / "src",
        openpi_root / "packages" / "openpi-client" / "src",
        libero_root / "src" / "LIBERO",
        libero_root,
    ]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    if libero_site_packages and libero_site_packages.is_dir():
        site_packages_str = str(libero_site_packages)
        if site_packages_str not in sys.path:
            sys.path.append(site_packages_str)


def ensure_runtime_env(libero_config_path: Path | None = None) -> None:
    if libero_config_path is not None:
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_config_path))
    os.environ.setdefault("MUJOCO_GL", "egl")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_distractor_assets(manifest_path: Path | None) -> list[dict[str, Any]]:
    if manifest_path is None or not manifest_path.exists():
        return []
    payload = load_json(manifest_path)
    manifest = payload.get("assets", payload)
    assets: list[dict[str, Any]] = []
    for entry in manifest:
        asset_path_value = entry.get("asset_path") or entry.get("filename")
        if asset_path_value is None:
            continue
        asset_path = Path(asset_path_value)
        if not asset_path.is_absolute():
            asset_path = manifest_path.parent / asset_path
        if not asset_path.exists():
            continue
        assets.append({"name": entry["name"], "image": Image.open(asset_path).convert("RGBA")})
    return assets


def with_alpha(image: Image.Image, alpha: int) -> Image.Image:
    rgba = image.copy()
    rgba.putalpha(alpha)
    return rgba


def paste_asset(canvas: Image.Image, asset: Image.Image, x: int, y: int, target_height: int, alpha: int) -> None:
    scale = target_height / asset.height
    target_width = max(1, int(round(asset.width * scale)))
    resized = asset.resize((target_width, target_height), Image.Resampling.LANCZOS)
    canvas.alpha_composite(with_alpha(resized, alpha), (x, y))


def apply_margin_noise_band(image: np.ndarray, shift_cfg: dict[str, Any], noise_seed: int) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    array = image.astype(np.int16)
    height, width = array.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    top_band = max(1, int(round(float(shift_cfg.get("top_fraction", 0.16)) * height)))
    side_band = max(1, int(round(float(shift_cfg.get("side_fraction", 0.10)) * width)))
    mask[:top_band, :] = True
    mask[:, :side_band] = True
    mask[:, width - side_band :] = True
    noise = rng.normal(0.0, float(shift_cfg.get("noise_std", 40.0)), size=array.shape)
    array[mask] = np.clip(array[mask] + noise[mask], 0, 255)
    return array.astype(np.uint8)


def apply_perimeter_clutter_noise_mix(
    image: np.ndarray,
    shift_cfg: dict[str, Any],
    noise_seed: int,
    distractor_assets: list[dict[str, Any]],
) -> np.ndarray:
    noised = apply_margin_noise_band(image, shift_cfg, noise_seed=noise_seed)
    if not distractor_assets:
        return noised

    pil = Image.fromarray(noised).convert("RGBA")
    width, height = pil.size
    target_height = max(1, int(round(float(shift_cfg.get("asset_target_height_fraction", 0.20)) * height)))
    alpha = int(shift_cfg.get("asset_alpha", 225))
    positions = [(0.03, 0.02), (0.24, 0.03), (0.78, 0.02)]
    for idx, (x_frac, y_frac) in enumerate(positions):
        asset = distractor_assets[(noise_seed + idx) % len(distractor_assets)]["image"]
        paste_asset(pil, asset, int(x_frac * width), int(y_frac * height), target_height, alpha)
    return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def apply_image_shift(
    image: np.ndarray,
    shift_cfg: dict[str, Any],
    noise_seed: int,
    distractor_assets: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    if not shift_cfg:
        return image
    if shift_cfg.get("kind") == "margin_noise_band":
        return apply_margin_noise_band(image, shift_cfg, noise_seed=noise_seed)
    if shift_cfg.get("kind") == "perimeter_clutter_noise_mix":
        return apply_perimeter_clutter_noise_mix(
            image,
            shift_cfg,
            noise_seed=noise_seed,
            distractor_assets=distractor_assets or [],
        )
    return image


def preprocess_image(
    obs: dict[str, np.ndarray],
    resize_size: int,
    shift_cfg: dict[str, Any],
    noise_seed: int,
    distractor_assets: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base = Image.fromarray(obs["agentview_image"][::-1, ::-1]).convert("RGB")
    wrist = Image.fromarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).convert("RGB")
    base_np = apply_image_shift(
        np.asarray(base, dtype=np.uint8),
        shift_cfg,
        noise_seed=noise_seed,
        distractor_assets=distractor_assets,
    )
    wrist_np = apply_image_shift(
        np.asarray(wrist, dtype=np.uint8),
        shift_cfg,
        noise_seed=noise_seed + 17,
        distractor_assets=distractor_assets,
    )
    base = Image.fromarray(base_np).convert("RGB")
    wrist = Image.fromarray(wrist_np).convert("RGB")
    base = base.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
    wrist = wrist.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
    return np.asarray(base, dtype=np.uint8), np.asarray(wrist, dtype=np.uint8)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    denominator = np.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator).astype(np.float32)


def get_dummy_action() -> list[float]:
    return [0.0] * 6 + [-1.0]


def get_max_steps(suite_name: str) -> int:
    if suite_name == "libero_spatial":
        return 220
    if suite_name == "libero_object":
        return 280
    if suite_name == "libero_goal":
        return 300
    if suite_name == "libero_10":
        return 520
    if suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown suite: {suite_name}")


def select_task_ids(task_suite: Any, requested_task_ids: list[int] | None, max_tasks: int | None) -> list[int]:
    if requested_task_ids:
        return requested_task_ids
    all_task_ids = list(range(task_suite.n_tasks))
    if max_tasks is not None:
        return all_task_ids[:max_tasks]
    return all_task_ids


def save_rollout_gif(frames: list[np.ndarray], output_path: Path, duration_ms: int) -> None:
    if not frames:
        return
    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB") for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def save_contact_sheet(frames: list[np.ndarray], output_path: Path, columns: int = 5) -> None:
    if not frames:
        return
    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB") for frame in frames]
    frame_w, frame_h = pil_frames[0].size
    columns = max(1, min(columns, len(pil_frames)))
    rows = math.ceil(len(pil_frames) / columns)
    canvas = Image.new("RGB", (columns * frame_w, rows * frame_h), color=(255, 255, 255))
    for idx, frame in enumerate(pil_frames):
        x = (idx % columns) * frame_w
        y = (idx // columns) * frame_h
        canvas.paste(frame, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def time_stamp() -> str:
    import datetime as dt

    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")
