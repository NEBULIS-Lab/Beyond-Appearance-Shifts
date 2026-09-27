#!/usr/bin/env python3
"""Record installed versions, external revisions and artifact hashes without model imports."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def named_path(value):
    name, separator, path = value.partition('=')
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError('expected LABEL=PATH')
    return name, Path(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=named_path, action='append', default=[])
    parser.add_argument('--artifact', type=named_path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for entries in (args.repo, args.artifact):
        if len({name for name, path in entries}) != len(entries):
            raise ValueError('labels must be unique within each input type')
    result = {'python': platform.python_version(), 'packages': {}, 'repositories': {}, 'artifacts': {}}
    for package in ('bas-vla', 'numpy', 'Pillow', 'torch', 'torchvision', 'transformers',
                    'huggingface_hub', 'imageio', 'jax', 'jaxlib', 'mujoco', 'robosuite'):
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = None
    for label, path in args.repo:
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip())
        result['repositories'][label] = {'revision': revision, 'dirty': dirty}
    for label, path in args.artifact:
        result['artifacts'][label] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
