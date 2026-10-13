"""Đọc ảnh và ground truth (định dạng label YOLO)."""

import os
from glob import glob

IMAGE_EXTS = ("*.jpg", "*.png")


def list_images(images_dir, max_images=None):
    """Danh sách ảnh đã sort, cùng thứ tự với legacy/dense_tiling_experiment.py."""
    paths = sorted(p for ext in IMAGE_EXTS
                   for p in glob(os.path.join(images_dir, ext)))
    return paths[:max_images] if max_images else paths


def label_path_for(image_path, labels_dir):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, stem + ".txt")


def load_gt(label_path, H, W):
    """Đọc label YOLO -> list (x1, y1, x2, y2, cls) theo pixel ảnh gốc."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                cls = int(float(p[0]))
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                              (cx + bw / 2) * W, (cy + bh / 2) * H, cls))
    return boxes


def list_research_images(images_dir, max_images=None):
    """Strict opt-in loader; legacy image ordering/extension behavior is unchanged."""
    from pathlib import Path
    paths = sorted(p for p in Path(images_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Image stems must be unique for one-to-one YOLO label mapping')
    return [str(p) for p in paths[:max_images]]


def load_gt_strict(label_path, height, width, nc=None):
    """Original-pixel boxes, clipped at image edges; malformed/missing labels fail."""
    from pathlib import Path
    import numpy as np
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        if not line.strip():
            continue
        row = list(map(float, line.split()))
        if len(row) != 5 or not np.isfinite(row).all():
            raise ValueError(f'Invalid YOLO label: {label_path}')
        cls, cx, cy, bw, bh = row
        if (cls != int(cls) or cls < 0 or (nc is not None and cls >= nc)
                or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1)):
            raise ValueError(f'Invalid YOLO class/coordinates: {label_path}')
        boxes.append((max(0, (cx-bw/2)*width), max(0, (cy-bh/2)*height),
                      min(width, (cx+bw/2)*width), min(height, (cy+bh/2)*height), int(cls)))
    return boxes


def check_class_map(data_yaml, model_names):
    """Check an optional explicit class map, never change detector class semantics."""
    from pathlib import Path
    import yaml
    if data_yaml is None:
        return
    declared = yaml.safe_load(Path(data_yaml).read_text())
    names = declared['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    actual = dict(enumerate(model_names)) if isinstance(model_names, list) else dict(model_names)
    if names != actual or declared.get('nc', len(names)) != len(names):
        raise ValueError('Dataset class map differs from detector checkpoint; supply a matched detector')
