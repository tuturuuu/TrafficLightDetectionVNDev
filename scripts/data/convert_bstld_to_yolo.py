#!/usr/bin/env python3
"""Convert Dataset Ninja BSTLD to YOLO detection without changing the source.

Usage: python scripts/data/convert_bstld_to_yolo.py
Images are hardlinked (or copied across filesystems). Validation holds out whole
training videos, choosing the subset closest to --val-fraction. The official
test set is preserved. Existing outputs are never overwritten.
"""

import argparse
from collections import Counter, defaultdict
import errno
import itertools
import json
import math
import os
from pathlib import Path
import shutil

from PIL import Image
import yaml


CLASS_NAMES = ("red", "yellow", "green", "off")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "data/BSTLD"


def yolo_box(obj, width, height):
    """Preserve the supplied xyxy corners, using the project's existing convention.

    Keep fractional centers and normalize by full image dimensions; adding one
    pixel or rounding the center would change the original corner coordinates.
    Invalid annotations fail explicitly instead of silently dropping objects.
    """
    if obj["geometryType"] != "rectangle":
        raise ValueError(f"Unsupported geometry: {obj['geometryType']}")
    class_id = CLASS_IDS[obj["classTitle"]]
    (xa, ya), (xb, yb) = obj["points"]["exterior"]
    x1, x2 = sorted((float(xa), float(xb)))
    y1, y2 = sorted((float(ya), float(yb)))
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Non-finite bounding box")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid box {(x1, y1, x2, y2)} for {width}x{height}")
    return (class_id, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def collect(source, split):
    image_dir, ann_dir = source / split / "img", source / split / "ann"
    images = sorted(p for p in image_dir.iterdir() if p.is_file())
    expected = {f"{image.name}.json" for image in images}
    if not images or expected != {p.name for p in ann_dir.glob("*.json")}:
        raise ValueError(f"Missing or unmatched images/annotations in {split}")
    samples, stems = [], set()
    for image in images:
        if image.stem in stems:
            raise ValueError(f"Duplicate label stem in {split}: {image.stem}")
        stems.add(image.stem)
        annotation = json.loads((ann_dir / f"{image.name}.json").read_text())
        width, height = annotation["size"]["width"], annotation["size"]["height"]
        with Image.open(image) as img:
            if img.size != (width, height):
                raise ValueError(f"Image/annotation size mismatch: {image}")
        groups = [tag["name"] for tag in annotation["tags"] if tag["name"].endswith("_bag")]
        if split == "train" and len(groups) != 1:
            raise ValueError(f"Expected exactly one training video tag: {image}")
        source_rows = [yolo_box(obj, width, height) for obj in annotation["objects"]]
        rows = list(dict.fromkeys(source_rows))
        samples.append({"image": image, "source_split": split,
                        "sequence": groups[0] if groups else None, "rows": rows,
                        "duplicates_removed": len(source_rows) - len(rows)})
    return samples


def validation_groups(samples, fraction):
    """Exhaustively choose among BSTLD's seven training videos; stable tie-break."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["sequence"]].append(sample)
    keys = sorted(groups)
    if not 2 <= len(keys) <= 20:
        raise ValueError("Expected 2–20 source videos for grouped validation")
    totals = Counter(row[0] for sample in samples for row in sample["rows"])
    candidates = []
    for count in range(1, len(keys)):
        for subset in itertools.combinations(keys, count):
            members = [sample for key in subset for sample in groups[key]]
            classes = Counter(row[0] for sample in members for row in sample["rows"])
            if any(classes[c] == 0 or classes[c] == totals[c] for c in CLASS_IDS.values()):
                continue
            candidates.append((abs(len(members) - len(samples) * fraction), subset))
    if not candidates:
        raise ValueError("Cannot retain every class in both train and validation")
    return set(min(candidates)[1])


def convert(source, output, fraction):
    source, output = source.resolve(), output.resolve()
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    meta = json.loads((source / "meta.json").read_text())
    if {item["title"] for item in meta["classes"]} != set(CLASS_NAMES):
        raise ValueError("Unexpected BSTLD classes")
    train, test = collect(source, "train"), collect(source, "test")
    held_out = validation_groups(train, fraction)
    partitions = {
        "train": [s for s in train if s["sequence"] not in held_out],
        "val": [s for s in train if s["sequence"] in held_out],
        "test": test,
    }
    staging = output.with_name(output.name + ".converting")
    staging.mkdir(parents=True, exist_ok=False)
    manifest, summary, link_counts = [], {}, Counter()
    for split, samples in partitions.items():
        image_dir, label_dir = staging / "images" / split, staging / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        counts = Counter()
        for sample in samples:
            image = sample["image"]
            try:
                os.link(image, image_dir / image.name)
                link_counts["hardlink"] += 1
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copy2(image, image_dir / image.name)
                link_counts["copy"] += 1
            lines = [f"{row[0]} " + " ".join(f"{v:.10f}" for v in row[1:])
                     for row in sample["rows"]]
            (label_dir / f"{image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts.update(CLASS_NAMES[row[0]] for row in sample["rows"])
            manifest.append({"split": split, "source_split": sample["source_split"],
                             "source_image": str(image.relative_to(source)),
                             "sequence": sample["sequence"],
                             "duplicates_removed": sample["duplicates_removed"]})
        summary[split] = {"images": len(samples), "labels": len(samples),
                          "background_images": sum(not s["rows"] for s in samples),
                          "duplicate_boxes_removed": sum(s["duplicates_removed"] for s in samples),
                          "objects": sum(counts.values()), "class_counts": dict(counts)}
        print(f"{split}: {summary[split]}", flush=True)
    config = {"path": str(output), "train": "images/train", "val": "images/val",
              "test": "images/test", "nc": len(CLASS_NAMES), "names": list(CLASS_NAMES)}
    (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    report = {"source": str(source), "output": str(output), "classes": CLASS_IDS,
              "val_fraction_requested": fraction,
              "val_fraction_actual": len(partitions["val"]) / len(train),
              "validation_sequences": sorted(held_out), "official_test_preserved": True,
              "train_val_sequence_overlap": [], "image_storage": dict(link_counts),
              "box_convention": "Original xyxy corners preserved; normalized cx, cy, width, height",
              "splits": summary}
    (staging / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (staging / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(output)
    print(f"YOLO dataset ready: {output / 'data.yaml'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, help="Default: SOURCE/yolo")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    convert(args.source, args.output or args.source / "yolo", args.val_fraction)


if __name__ == "__main__":
    main()
