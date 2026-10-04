"""
Cắt dataset thành tile uniform (offline) bằng tay. scripts/train_detector.py
đã tự làm việc này khi tiling.type=uniform; dùng script này khi cần cắt riêng
(vd tập test).

    python scripts/tile_dataset.py --dataset-path data/kayuan/data.yaml --splits test
"""

import os
import cv2
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm


dataset_path = "data/SeaDroneSees_70_15_15/data.yaml"   # chạy từ gốc repo
splits = ["test"]
tile_size = 640
overlap = 0.2
min_visibility = 0.3
output_suffix = "_tiled_uniform"
# ============================


def resolve_dataset(dataset_path_arg):
    dataset_path = Path(dataset_path_arg).expanduser()
    if not dataset_path.is_file():
        return dataset_path, {}

    with dataset_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    root = Path(data.get("path", dataset_path.parent)).expanduser()
    if not root.is_absolute():
        root = dataset_path.parent / root
    return root, data


def resolve_split_image_dir(dataset_root, dataset_config, split):
    split_path = dataset_config.get(split)
    if split_path is None:
        return dataset_root / "images" / split
    if not isinstance(split_path, str):
        raise ValueError(f"Expected {split!r} in data.yaml to be a directory path.")

    split_path = Path(split_path).expanduser()
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    return split_path


def resolve_label_dir(dataset_root, img_dir, split):
    try:
        parts = list(img_dir.relative_to(dataset_root).parts)
    except ValueError:
        parts = []

    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return dataset_root.joinpath(*parts)

    return dataset_root / "labels" / split


# Hàm cắt tile nằm trong adaptile (dùng chung với scripts/train_detector.py).
from adaptile.dataset_tiling import tile_image_and_labels  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default=dataset_path,
        help="Dataset root directory or YOLO data.yaml path.",
    )
    parser.add_argument("--splits", nargs="+", default=splits)
    parser.add_argument("--tile-size", type=int, default=tile_size)
    parser.add_argument("--overlap", type=float, default=overlap)
    parser.add_argument("--min-visibility", type=float, default=min_visibility)
    parser.add_argument(
        "--output-suffix",
        default=output_suffix,
        help="Output folders are named {split}{output_suffix}/images|labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into non-empty output directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root, dataset_config = resolve_dataset(args.dataset_path)

    for split in args.splits:
        img_dir = resolve_split_image_dir(dataset_root, dataset_config, split)
        lbl_dir = resolve_label_dir(dataset_root, img_dir, split)
        out_img_dir = dataset_root / f"{split}{args.output_suffix}" / "images"
        out_lbl_dir = dataset_root / f"{split}{args.output_suffix}" / "labels"

        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not lbl_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

        for out_dir in (out_img_dir, out_lbl_dir):
            if out_dir.is_dir() and any(out_dir.iterdir()) and not args.overwrite:
                raise RuntimeError(
                    f"Output directory is not empty: {out_dir}\n"
                    "Use a different --output-suffix or pass --overwrite explicitly."
                )

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            f.name for f in img_dir.iterdir()
            if f.name.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        print(f"\n📸 Tiling {len(image_files)} {split} images...")
        print(f"   images -> {out_img_dir}")
        print(f"   labels -> {out_lbl_dir}")

        for f in tqdm(image_files):
            img_path = img_dir / f
            lbl_path = lbl_dir / (Path(f).stem + ".txt")
            tile_image_and_labels(
                str(img_path), str(lbl_path), str(out_img_dir), str(out_lbl_dir),
                args.tile_size, args.overlap, args.min_visibility
            )

    print("✅ Done! Tiled dataset ready.")


if __name__ == "__main__":
    main()
