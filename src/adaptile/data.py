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
