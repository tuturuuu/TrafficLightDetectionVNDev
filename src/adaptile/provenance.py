"""Fingerprints used to pair LTS heads, detectors and experiment records."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def architecture_sha256(model):
    keys = ('nc', 'backbone', 'head', 'scales', 'scale', 'reg_max', 'end2end', 'channels')
    definition = {k: model.yaml[k] for k in keys if k in model.yaml}
    return hashlib.sha256(json.dumps(definition, sort_keys=True, default=str).encode()).hexdigest()


def software_versions():
    versions = {}
    for name in ('torch', 'torchvision', 'ultralytics', 'numpy', 'pycocotools', 'opencv-python'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_fingerprints():
    root = Path(__file__).resolve().parents[2]
    files = list((root / 'src/adaptile').rglob('*.py')) + list((root / 'scripts').glob('*.py'))
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(files)}
