from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenVLAOFTRuntime:
    repo_root: Path
    checkpoint_path: Path
    datasets_root: Path | None
    libero_config_path: Path | None
    libero_root: Path | None
    site_packages: Path | None
    lerobot_site_packages: Path | None
    python_bin: str


def _optional_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required path: set {name}")
    return Path(value)


def resolve_openvla_oft_runtime() -> OpenVLAOFTRuntime:
    return OpenVLAOFTRuntime(
        repo_root=_required_env_path("BAS_OPENVLA_OFT_ROOT"),
        checkpoint_path=_required_env_path("BAS_OPENVLA_OFT_CHECKPOINT"),
        datasets_root=_optional_env_path("BAS_LIBERO_DATASETS_ROOT"),
        libero_config_path=_optional_env_path("LIBERO_CONFIG_PATH"),
        libero_root=_optional_env_path("BAS_LIBERO_ROOT"),
        site_packages=_optional_env_path("BAS_OPENVLA_OFT_SITE_PACKAGES"),
        lerobot_site_packages=_optional_env_path("BAS_LEROBOT_SITE_PACKAGES"),
        python_bin=os.environ.get("BAS_PYTHON_BIN", "python3"),
    )


def build_semantic_break_command(
    *,
    python_bin: str,
    runner: Path,
    output_dir: Path,
    pairs_config: Path,
    pair_id: str | None = None,
    num_trials: int = 5,
    seed: int = 7,
) -> list[str]:
    command = [
        python_bin,
        str(runner),
        "--pairs-config",
        str(pairs_config),
        "--output-dir",
        str(output_dir),
        "--num-trials",
        str(num_trials),
        "--seed",
        str(seed),
    ]
    if pair_id:
        command.extend(["--pair-id", pair_id])
    return command
