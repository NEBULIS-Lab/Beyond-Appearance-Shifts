from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_video_frames(video_path: str | Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def sample_frame_indices(num_available: int, num_requested: int) -> list[int]:
    if num_available <= 0:
        raise ValueError("video contains no frames")
    if num_requested <= 1:
        return [0]
    indices = np.linspace(0, num_available - 1, num=num_requested)
    return [int(round(idx)) for idx in indices]


def build_video_montage(
    frames: list[np.ndarray],
    indices: list[int],
    *,
    columns: int = 5,
    tile_padding: int = 8,
    header_text: str = "",
) -> Image.Image:
    first = Image.fromarray(frames[indices[0]])
    tile_w, tile_h = first.size
    rows = int(np.ceil(len(indices) / columns))
    label_h = 18
    header_h = 28 if header_text else 0
    canvas_w = columns * tile_w + (columns + 1) * tile_padding
    canvas_h = header_h + rows * (tile_h + label_h) + (rows + 1) * tile_padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 248, 243))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    if header_text:
        draw.text((tile_padding, 8), header_text, fill=(20, 20, 20), font=font)

    for slot, frame_idx in enumerate(indices):
        row = slot // columns
        col = slot % columns
        x = tile_padding + col * (tile_w + tile_padding)
        y = header_h + tile_padding + row * (tile_h + label_h + tile_padding)
        canvas.paste(Image.fromarray(frames[frame_idx]), (x, y))
        draw.text((x, y + tile_h + 2), f"frame {frame_idx:03d}", fill=(40, 40, 40), font=font)
    return canvas
