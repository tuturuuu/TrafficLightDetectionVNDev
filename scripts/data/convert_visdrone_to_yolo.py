#!/usr/bin/env python3
"""Convert original VisDrone DET TXT annotations for images already split on disk.

python scripts/data/convert_visdrone_to_yolo.py --dataset-root data/visdrone

Input columns: left, top, width, height, score, category, truncation, occlusion.
Ignore score=0 and categories 0/11; map categories 1..10 to YOLO 0..9.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]
RAW_SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val",
              "test": "VisDrone2019-DET-test-dev"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_annotation(text, width, height):
    """Return YOLO text and auditable counts; reject malformed annotation rows."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    lines, stats = [], Counter()
    classes = Counter()
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        columns = raw.strip().rstrip(",").split(",")
        if len(columns) != 8:
            raise ValueError(f"Line {number}: expected 8 columns, got {len(columns)}")
        values = list(map(float, columns))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Line {number}: non-finite value")
        x, y, w, h, score, category, _, _ = values
        if not category.is_integer() or not 0 <= category <= 11 or score not in (0, 1):
            raise ValueError(f"Line {number}: invalid category/ground-truth score")
        stats["source_boxes"] += 1
        if score == 0 or category in (0, 11):
            stats["ignored_boxes"] += 1
            continue
        x1, y1 = max(0.0, min(x, width)), max(0.0, min(y, height))
        x2, y2 = max(0.0, min(x + w, width)), max(0.0, min(y + h, height))
        if x2 <= x1 or y2 <= y1:
            stats["invalid_boxes"] += 1
            continue
        if (x1, y1, x2, y2) != (x, y, x + w, y + h):
            stats["clipped_boxes"] += 1
        cls = int(category) - 1
        lines.append(f"{cls} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} "
                     f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n")
        classes[cls] += 1
        stats["written_boxes"] += 1
    return "".join(lines), stats, classes


def convert_dataset(root, overwrite=False):
    """Validate all inputs/outputs before writing labels, YAML and the report."""
    root = Path(root).resolve()
    pending, report = {}, {"format": "VisDrone DET TXT -> YOLO", "splits": {}}
    for split, source in RAW_SPLITS.items():
        image_dir = root / "images" / split
        annotation_dir = root / "raw" / source / "annotations"
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No images in {image_dir}")
        stems = {p.stem for p in images}
        if len(stems) != len(images):
            raise ValueError(f"Duplicate image stems in {image_dir}")
        annotations = {p.stem for p in annotation_dir.glob("*.txt")}
        if stems != annotations:
            raise ValueError(f"{split}: missing annotations={len(stems-annotations)}, "
                             f"annotations without images={len(annotations-stems)}")
        label_dir = root / "labels" / split
        stale = {p.stem for p in label_dir.glob("*.txt")} - stems
        if stale:
            raise ValueError(f"{split}: labels without images: {sorted(stale)[:5]}")
        totals = Counter(images=len(images), empty_labels=0, source_boxes=0,
                         written_boxes=0, ignored_boxes=0, invalid_boxes=0, clipped_boxes=0)
        classes = Counter()
        for image in images:
            annotation = annotation_dir / (image.stem + ".txt")
            with Image.open(image) as img:
                width, height = img.size
            try:
                text, stats, counts = convert_annotation(annotation.read_text(), width, height)
            except ValueError as exc:
                raise ValueError(f"{annotation}: {exc}") from exc
            pending[label_dir / annotation.name] = text
            totals.update(stats)
            totals["empty_labels"] += not text
            classes.update(counts)
        report["splits"][split] = {**totals, "class_counts": dict(sorted(classes.items()))}
    data = {"path": str(root), **{s: f"images/{s}" for s in RAW_SPLITS},
            "nc": len(NAMES), "names": dict(enumerate(NAMES))}
    pending[root / "data.yaml"] = yaml.safe_dump(data, sort_keys=False)
    pending[root / "conversion_report.json"] = json.dumps(report, indent=2) + "\n"
    for path, text in pending.items():
        if path.exists() and path.read_text() != text and not overwrite:
            raise FileExistsError(f"Different output already exists: {path}; use --overwrite")
    for path, text in pending.items():
        if not path.exists() or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(convert_dataset(args.dataset_root, args.overwrite), indent=2))


if __name__ == "__main__":
    main()
