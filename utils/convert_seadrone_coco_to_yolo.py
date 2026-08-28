#!/usr/bin/env python3
"""Convert SeaDroneSees COCO annotations to YOLO label files.

Default behavior skips the COCO category named "ignored" and remaps the
remaining category ids to contiguous YOLO class ids.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="data/SeaDroneSees",
        help="SeaDroneSees root with images/ and annotations/ directories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to convert. Expects annotations/instances_<split>.json.",
    )
    parser.add_argument(
        "--skip-category",
        default="ignored",
        help="Category name to skip. Use an empty string to keep all categories.",
    )
    parser.add_argument(
        "--yaml-name",
        default="data.yaml",
        help="Output YOLO dataset yaml filename under dataset root.",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_map(coco: dict, skip_category: str) -> tuple[dict[int, int], list[str]]:
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    kept = [
        c for c in categories
        if not skip_category or c["name"] != skip_category
    ]
    cat_id_to_yolo = {c["id"]: idx for idx, c in enumerate(kept)}
    names = [c["name"] for c in kept]
    return cat_id_to_yolo, names


def coco_bbox_to_yolo_line(
    category_id: int,
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> str | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or image_width <= 0 or image_height <= 0:
        return None

    x1 = max(0.0, min(float(x), float(image_width)))
    y1 = max(0.0, min(float(y), float(image_height)))
    x2 = max(0.0, min(float(x + w), float(image_width)))
    y2 = max(0.0, min(float(y + h), float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    bw = (x2 - x1) / image_width
    bh = (y2 - y1) / image_height
    return f"{category_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_split(
    dataset_root: Path,
    split: str,
    cat_id_to_yolo: dict[int, int],
) -> dict:
    ann_path = dataset_root / "annotations" / f"instances_{split}.json"
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    coco = load_coco(ann_path)
    images = {img["id"]: img for img in coco["images"]}
    labels_by_image: dict[int, list[str]] = defaultdict(list)
    skipped_categories = Counter()
    class_counts = Counter()
    invalid_boxes = 0

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in cat_id_to_yolo:
            skipped_categories[cat_id] += 1
            continue

        image = images.get(ann["image_id"])
        if image is None:
            invalid_boxes += 1
            continue

        yolo_id = cat_id_to_yolo[cat_id]
        line = coco_bbox_to_yolo_line(
            yolo_id,
            ann["bbox"],
            int(image["width"]),
            int(image["height"]),
        )
        if line is None:
            invalid_boxes += 1
            continue

        labels_by_image[ann["image_id"]].append(line)
        class_counts[yolo_id] += 1

    label_dir.mkdir(parents=True, exist_ok=True)
    missing_images = 0
    for image in coco["images"]:
        image_path = image_dir / image["file_name"]
        if not image_path.exists():
            missing_images += 1
        label_path = label_dir / f"{Path(image['file_name']).stem}.txt"
        lines = labels_by_image.get(image["id"], [])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "images": len(coco["images"]),
        "annotations_written": sum(class_counts.values()),
        "empty_label_files": len(coco["images"]) - len(labels_by_image),
        "missing_images": missing_images,
        "invalid_boxes": invalid_boxes,
        "skipped_categories": dict(skipped_categories),
        "class_counts": dict(sorted(class_counts.items())),
    }


def write_data_yaml(dataset_root: Path, yaml_name: str, names: list[str]) -> Path:
    yaml_path = dataset_root / yaml_name
    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if (dataset_root / "images" / "test").exists():
        lines.append("test: images/test")
    lines.extend(["", "names:"])
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    first_coco = load_coco(dataset_root / "annotations" / f"instances_{args.splits[0]}.json")
    cat_id_to_yolo, names = build_category_map(first_coco, args.skip_category)

    print("Category mapping:")
    for coco_id, yolo_id in sorted(cat_id_to_yolo.items()):
        print(f"  COCO {coco_id} -> YOLO {yolo_id} ({names[yolo_id]})")

    for split in args.splits:
        stats = convert_split(dataset_root, split, cat_id_to_yolo)
        print(f"\n{split}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    yaml_path = write_data_yaml(dataset_root, args.yaml_name, names)
    print(f"\nWrote {yaml_path}")


if __name__ == "__main__":
    main()
