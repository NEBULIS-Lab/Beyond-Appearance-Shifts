from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_episode_frames(episode_dir: str | Path) -> list[Path]:
    frame_dir = Path(episode_dir)
    frames = sorted(frame_dir.glob("frame_*.png"))
    if frames:
        return frames
    final_frame = frame_dir / "final_frame.png"
    return [final_frame] if final_frame.is_file() else []


def _read_label(frame_path: Path) -> str:
    stem = frame_path.stem
    if "_step_" in stem:
        return f"step {stem.rsplit('_step_', 1)[-1]}"
    return stem


def build_pair_montage(reference_paths: list[Path], variant_paths: list[Path], title: str) -> Image.Image:
    if not reference_paths or not variant_paths:
        raise ValueError("both episode directories must contain at least one frame")

    pair_count = min(len(reference_paths), len(variant_paths))
    reference_paths = reference_paths[:pair_count]
    variant_paths = variant_paths[:pair_count]

    first = Image.open(reference_paths[0]).convert("RGB")
    tile_w, tile_h = first.size
    title_h = 36
    label_h = 18
    col_gap = 12
    row_gap = 10
    margin = 16

    width = margin * 2 + tile_w * 2 + col_gap
    height = margin * 2 + title_h + pair_count * (tile_h + label_h) + (pair_count - 1) * row_gap

    canvas = Image.new("RGB", (width, height), color=(250, 248, 244))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 18), "reference", fill=(40, 40, 40), font=font)
    draw.text((margin + tile_w + col_gap, margin + 18), "variant", fill=(40, 40, 40), font=font)

    y = margin + title_h
    for ref_path, var_path in zip(reference_paths, variant_paths):
        ref_img = Image.open(ref_path).convert("RGB")
        var_img = Image.open(var_path).convert("RGB")
        canvas.paste(ref_img, (margin, y))
        canvas.paste(var_img, (margin + tile_w + col_gap, y))
        label_y = y + tile_h + 2
        draw.text((margin, label_y), _read_label(ref_path), fill=(70, 70, 70), font=font)
        draw.text((margin + tile_w + col_gap, label_y), _read_label(var_path), fill=(70, 70, 70), font=font)
        y += tile_h + label_h + row_gap
    return canvas
