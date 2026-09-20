#!/usr/bin/env python3
from __future__ import annotations

import os
import importlib.util
from pathlib import Path


def has_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


CHECKS = {
    "openpi": [
        "BAS_OPENPI_ROOT",
        "BAS_LIBERO_ROOT",
        "BAS_OPENPI_CHECKPOINT",
    ],
    "openvla_oft": [
        "BAS_OPENVLA_OFT_ROOT",
        "BAS_OPENVLA_OFT_CHECKPOINT",
    ],
    "grounded_preserving": [
        "BAS_GROUNDING_DINO_MODEL_ID",
        "BAS_SAM2_MODEL_ID",
        "BAS_GROUNDED_DEVICE",
    ],
}


def describe(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        return f"{name}: missing"
    if name == "BAS_GROUNDED_DEVICE":
        return f"{name}: {value} [configured device]"
    if name.endswith("MODEL_ID") and not value.startswith(("/", "./", "../")):
        return f"{name}: {value} [model identifier]"
    path = Path(value)
    status = "exists" if path.exists() else "missing-path"
    return f"{name}: {path} [{status}]"


def main() -> int:
    print(f"[python]")
    print(f"transformers: {'available' if has_transformers() else 'missing'}")
    print()
    for section, names in CHECKS.items():
        print(f"[{section}]")
        for name in names:
            print(describe(name))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
