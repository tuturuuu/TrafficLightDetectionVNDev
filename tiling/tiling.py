import os
import cv2
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm


dataset_path = "/home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev/data/SeaDroneSees_70_15_15/data.yaml"
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


def tile_image_and_labels(
    img_path, label_path, out_img_dir, out_lbl_dir,
    tile_size=480, overlap=0.2, min_visibility=0.3
):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Skipping unreadable image: {img_path}")
        return

    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap))
    base = Path(img_path).stem

    # --- Read YOLO labels (pixel coords) ---
    labels = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                cx_px = xc * w
                cy_px = yc * h
                bw_px = bw * w
                bh_px = bh * h
                labels.append((
                    int(cls),
                    cx_px - bw_px / 2,
                    cy_px - bh_px / 2,
                    cx_px + bw_px / 2,
                    cy_px + bh_px / 2,
                    bw_px * bh_px,
                ))

    # --- Tile start positions that guarantee full coverage ---
    def get_starts(length):
        if length <= tile_size:
            return [0]                          # single tile, will be padded if needed
        starts = list(range(0, length - tile_size + 1, step))
        last_valid = length - tile_size
        if starts[-1] < last_valid:             # edge strip not yet covered
            starts.append(last_valid)
        return starts

    tile_id = 0
    for ty in get_starts(h):
        for tx in get_starts(w):
            tx2, ty2 = tx + tile_size, ty + tile_size

            tile = img[ty:ty2, tx:tx2]

            # Pad only when image is smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = cv2.copyMakeBorder(
                    tile,
                    0, tile_size - tile.shape[0],
                    0, tile_size - tile.shape[1],
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )

            cv2.imwrite(os.path.join(out_img_dir, f"{base}_tile{tile_id}.jpg"), tile)

            out_lbl = os.path.join(out_lbl_dir, f"{base}_tile{tile_id}.txt")
            with open(out_lbl, "w") as lf:
                for cls, bx1, by1, bx2, by2, orig_area in labels:
                    ix1 = max(bx1, tx);  iy1 = max(by1, ty)
                    ix2 = min(bx2, tx2); iy2 = min(by2, ty2)

                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    clipped_area = (ix2 - ix1) * (iy2 - iy1)
                    if orig_area > 0 and (clipped_area / orig_area) < min_visibility:
                        continue

                    cx_n = max(0.0, min(1.0, ((ix1 + ix2) / 2 - tx) / tile_size))
                    cy_n = max(0.0, min(1.0, ((iy1 + iy2) / 2 - ty) / tile_size))
                    bw_n = max(0.0, min(1.0, (ix2 - ix1) / tile_size))
                    bh_n = max(0.0, min(1.0, (iy2 - iy1) / tile_size))

                    if bw_n > 0 and bh_n > 0:
                        lf.write(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")

            tile_id += 1


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
