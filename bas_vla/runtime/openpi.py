from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def require_path(path: Path | None, arg_name: str, env_name: str) -> Path:
    if path is None:
        raise SystemExit(f"missing required path: set {env_name} or pass {arg_name}")
    return path
