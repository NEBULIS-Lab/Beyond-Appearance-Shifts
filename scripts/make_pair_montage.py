#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bas_vla.media.image_montage import build_pair_montage, load_episode_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reference-vs-variant montage from saved episode frames.")
    parser.add_argument("--reference-episode-dir", required=True)
    parser.add_argument("--variant-episode-dir", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--title", default="reference vs variant")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_image).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage = build_pair_montage(
        load_episode_frames(args.reference_episode_dir),
        load_episode_frames(args.variant_episode_dir),
        args.title,
    )
    montage.save(output_path)
    print(f"saved montage to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
