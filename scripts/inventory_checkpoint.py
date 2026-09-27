#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib

def main():
    parser = argparse.ArgumentParser(description="Hash adapter artifacts without loading weights or initializing a model.")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = []
    for name in ("residual_adapter.pt", "adapter_metadata.json", "vocab.json", "results.json"):
        path = args.artifact_dir / name
        if not path.is_file():
            parser.error(f"required artifact missing: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"filename": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    metadata = json.loads((args.artifact_dir / "adapter_metadata.json").read_text())
    required = ("input_dim", "output_dim", "hidden_dim", "delta_scale", "vocab_size", "feature_type", "feature_normalization", "objective_config", "seed", "action_space", "action_layout", "carrier", "action_normalization")
    missing = [key for key in required if key not in metadata]
    if missing:
        parser.error(f"metadata missing required fields: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "status": "locally_supplied", "metadata": metadata, "artifacts": entries}, indent=2) + "\n")

if __name__ == "__main__":
    main()
