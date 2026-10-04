#!/usr/bin/env python3
"""Create a labeled 70/15/15 SeaDroneSees split from original train+val."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_SPLITS = ("train", "val")
TARGET_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    source_split: str

    @property
    def file_name(self) -> str:
        return self.image_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the labeled SeaDroneSees train+val data into new "
            "train/val/test sets. The original test images are ignored because "
            "they do not have labels/annotations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/SeaDroneSees"),
        help="Dataset root containing images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/SeaDroneSees_70_15_15"),
        help="Destination dataset root. The source dataset is left untouched.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place images and labels in the output dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before recreating it.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r <= 0 for r in ratios):
        raise ValueError("All split ratios must be positive.")

    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


def collect_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen_names: dict[str, Path] = {}

    for split in SOURCE_SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if image_path.name in seen_names:
                raise ValueError(
                    "Duplicate image file name across train/val: "
                    f"{image_path.name} ({seen_names[image_path.name]} and {image_path})"
                )
            seen_names[image_path.name] = image_path

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

            samples.append(Sample(image_path=image_path, label_path=label_path, source_split=split))

    if not samples:
        raise ValueError(f"No labeled train/val images found in {dataset_root}")
    return samples


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = round(total * train_ratio)
    val_count = round(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Use --overwrite to recreate it, or pass a different --output-root."
            )
        shutil.rmtree(output_root)

    for split in TARGET_SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "annotations").mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")


def copy_split_files(splits: dict[str, list[Sample]], output_root: Path, copy_mode: str) -> None:
    for split, samples in splits.items():
        for sample in samples:
            place_file(sample.image_path, output_root / "images" / split / sample.file_name, copy_mode)
            place_file(sample.label_path, output_root / "labels" / split / sample.label_path.name, copy_mode)


def load_coco_annotations(dataset_root: Path) -> tuple[dict[str, dict], dict[int, list[dict]], dict[str, object]]:
    images_by_file_name: dict[str, dict] = {}
    annotations_by_image_id: dict[int, list[dict]] = {}
    metadata: dict[str, object] | None = None

    for split in SOURCE_SPLITS:
        annotation_path = dataset_root / "annotations" / f"instances_{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        if metadata is None:
            metadata = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "categories": data.get("categories", []),
            }

        image_ids_in_file: set[int] = set()
        for image in data.get("images", []):
            file_name = image["file_name"]
            if file_name in images_by_file_name:
                raise ValueError(f"Duplicate COCO image file_name: {file_name}")
            images_by_file_name[file_name] = image
            image_ids_in_file.add(image["id"])

        for annotation in data.get("annotations", []):
            image_id = annotation["image_id"]
            if image_id in image_ids_in_file:
                annotations_by_image_id.setdefault(image_id, []).append(annotation)

    if metadata is None:
        raise ValueError("No COCO annotations were loaded.")

    return images_by_file_name, annotations_by_image_id, metadata


def write_coco_split_files(dataset_root: Path, output_root: Path, splits: dict[str, list[Sample]]) -> None:
    images_by_file_name, annotations_by_image_id, metadata = load_coco_annotations(dataset_root)

    for split, samples in splits.items():
        output_images: list[dict] = []
        output_annotations: list[dict] = []
        old_to_new_image_id: dict[int, int] = {}

        for new_image_id, sample in enumerate(samples, start=1):
            source_image = images_by_file_name.get(sample.file_name)
            if source_image is None:
                raise KeyError(f"{sample.file_name} is missing from train/val COCO annotations.")

            image = dict(source_image)
            old_image_id = image["id"]
            image["id"] = new_image_id
            output_images.append(image)
            old_to_new_image_id[old_image_id] = new_image_id

        next_annotation_id = 1
        for sample in samples:
            old_image_id = images_by_file_name[sample.file_name]["id"]
            for source_annotation in annotations_by_image_id.get(old_image_id, []):
                annotation = dict(source_annotation)
                annotation["id"] = next_annotation_id
                annotation["image_id"] = old_to_new_image_id[old_image_id]
                output_annotations.append(annotation)
                next_annotation_id += 1

        output = {
            "info": metadata["info"],
            "licenses": metadata["licenses"],
            "categories": metadata["categories"],
            "images": output_images,
            "annotations": output_annotations,
        }
        annotation_path = output_root / "annotations" / f"instances_{split}.json"
        annotation_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


def parse_names_from_data_yaml(data_yaml: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    in_names = False

    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t")):
                break
            if ":" not in stripped:
                continue
            idx, name = stripped.split(":", 1)
            names[int(idx.strip())] = name.strip().strip("'\"")

    return names


def names_from_coco_categories(dataset_root: Path) -> dict[int, str]:
    annotation_path = dataset_root / "annotations" / "instances_train.json"
    if not annotation_path.is_file():
        return {}

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    names: dict[int, str] = {}
    for idx, category in enumerate(data.get("categories", [])):
        category_id = category.get("id", idx)
        names[int(category_id)] = str(category["name"])
    return names


def write_data_yaml(dataset_root: Path, output_root: Path) -> None:
    data_yaml = dataset_root / "data.yaml"
    names = parse_names_from_data_yaml(data_yaml) if data_yaml.is_file() else {}
    if not names:
        names = names_from_coco_categories(dataset_root)
    if not names:
        raise ValueError("Could not infer class names from data.yaml or COCO categories.")

    names_block = "\n".join(f"  {idx}: {name}" for idx, name in sorted(names.items()))
    output_yaml = "\n".join(
        [
            f"path: {output_root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            names_block,
            "",
        ]
    )
    (output_root / "data.yaml").write_text(output_yaml, encoding="utf-8")


def print_summary(splits: dict[str, list[Sample]], output_root: Path) -> None:
    total = sum(len(samples) for samples in splits.values())
    print("Split summary")
    for split in TARGET_SPLITS:
        count = len(splits[split])
        print(f"  {split:5s}: {count:5d} images ({count / total:.2%})")
    print(f"\nOutput dataset: {output_root}")
    print(f"YOLO config:    {output_root / 'data.yaml'}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    dataset_root = resolve(args.dataset_root)
    output_root = resolve(args.output_root)

    samples = collect_samples(dataset_root)
    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)

    prepare_output_root(output_root, args.overwrite)
    copy_split_files(splits, output_root, args.copy_mode)
    write_coco_split_files(dataset_root, output_root, splits)
    write_data_yaml(dataset_root, output_root)
    print_summary(splits, output_root)


if __name__ == "__main__":
    main()
