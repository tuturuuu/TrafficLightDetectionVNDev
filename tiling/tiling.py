import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


dataset_path = "/home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev/data/SeaDroneSees_70_15_15/data.yaml"
splits = ["train", "val"]
tile_size = 640
overlap = 0.2
min_visibility = 0.3
output_suffix = "_tiled_uniform"
# ============================

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
    parser.add_argument("--dataset-path", default=dataset_path)
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

    for split in args.splits:
        img_dir = os.path.join(args.dataset_path, "images", split)
        lbl_dir = os.path.join(args.dataset_path, "labels", split)
        out_img_dir = os.path.join(args.dataset_path, f"{split}{args.output_suffix}", "images")
        out_lbl_dir = os.path.join(args.dataset_path, f"{split}{args.output_suffix}", "labels")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(lbl_dir):
            raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

        for out_dir in (out_img_dir, out_lbl_dir):
            if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.overwrite:
                raise RuntimeError(
                    f"Output directory is not empty: {out_dir}\n"
                    "Use a different --output-suffix or pass --overwrite explicitly."
                )

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        print(f"\n📸 Tiling {len(image_files)} {split} images...")
        print(f"   images -> {out_img_dir}")
        print(f"   labels -> {out_lbl_dir}")

        for f in tqdm(image_files):
            img_path = os.path.join(img_dir, f)
            lbl_path = os.path.join(lbl_dir, Path(f).stem + ".txt")
            tile_image_and_labels(
                img_path, lbl_path, out_img_dir, out_lbl_dir,
                args.tile_size, args.overlap, args.min_visibility
            )

    print("✅ Done! Tiled dataset ready.")


if __name__ == "__main__":
    main()
