#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.media.video_montage import build_video_montage, load_video_frames, sample_frame_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample frames from a rollout video and build a montage.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-padding", type=int, default=8)
    parser.add_argument("--header-text", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_image = Path(args.output_image).resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    frames = load_video_frames(args.video_path)
    indices = sample_frame_indices(len(frames), args.num_frames)
    montage = build_video_montage(
        frames,
        indices,
        columns=args.columns,
        tile_padding=args.tile_padding,
        header_text=args.header_text,
    )
    montage.save(output_image)
    print(f"saved montage to {output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
