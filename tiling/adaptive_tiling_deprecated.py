import os
import cv2
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ========== CONFIG ==========
dataset_path = "/home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev/data/SeaDroneSees"
splits = ["train", "val"]     # SeaDroneSees test has no labels here
# Keep the materialized dataset bounded. For SeaDroneSee this targets grids
# such as 2x2, 2x3, 3x2, 2x4, or 4x2 instead of dense sliding windows.
min_tiles_per_image = 4
max_tiles_per_image = 8

target_nba = 0.04
grid_overlap = 0.05
min_visibility = 0.20
jpeg_quality = 80
# ============================


def compute_nba(labels, img_w, img_h):
    if not labels:
        return None

    img_area = img_w * img_h
    nbas = []
    for _, x1, y1, x2, y2 in labels:
        box_area = (x2 - x1) * (y2 - y1)
        nbas.append(box_area / img_area)

    return float(np.percentile(nbas, 10))


def compute_desired_tile_count(nba):
    if nba is None or nba <= 0:
        return min_tiles_per_image

    n_axis = math.ceil(math.sqrt(target_nba / nba))
    desired = n_axis * n_axis
    return max(min_tiles_per_image, min(desired, max_tiles_per_image))


def choose_grid_shape(img_w, img_h, desired_tiles):
    candidates = []
    for rows in range(1, max_tiles_per_image + 1):
        for cols in range(1, max_tiles_per_image + 1):
            total = rows * cols
            if min_tiles_per_image <= total <= max_tiles_per_image:
                candidates.append((rows, cols, total))

    image_aspect = img_w / max(img_h, 1)

    def score(candidate):
        rows, cols, total = candidate
        tile_aspect = (img_w / cols) / max(img_h / rows, 1)
        square_penalty = abs(math.log(max(tile_aspect, 1e-6)))
        count_penalty = abs(total - desired_tiles) * 0.35

        orientation_penalty = 0.0
        if image_aspect >= 1 and cols < rows:
            orientation_penalty = 0.5
        elif image_aspect < 1 and rows < cols:
            orientation_penalty = 0.5

        return square_penalty + count_penalty + orientation_penalty

    return min(candidates, key=score)[:2]


def make_grid_boxes(img_w, img_h, rows, cols, overlap_ratio):
    x_edges = np.linspace(0, img_w, cols + 1).round().astype(int)
    y_edges = np.linspace(0, img_h, rows + 1).round().astype(int)

    boxes = []
    for row in range(rows):
        for col in range(cols):
            x1 = x_edges[col]
            x2 = x_edges[col + 1]
            y1 = y_edges[row]
            y2 = y_edges[row + 1]

            pad_x = int((x2 - x1) * overlap_ratio)
            pad_y = int((y2 - y1) * overlap_ratio)

            boxes.append((
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(img_w, x2 + pad_x),
                min(img_h, y2 + pad_y),
            ))

    return boxes


def read_yolo_labels(label_path, img_w, img_h):
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)
            cx_px = xc * img_w
            cy_px = yc * img_h
            bw_px = bw * img_w
            bh_px = bh * img_h
            labels.append((
                int(cls),
                cx_px - bw_px / 2,
                cy_px - bh_px / 2,
                cx_px + bw_px / 2,
                cy_px + bh_px / 2,
            ))

    return labels


def write_tile_labels(label_file, labels, tile_box):
    tx1, ty1, tx2, ty2 = tile_box
    tile_w = tx2 - tx1
    tile_h = ty2 - ty1

    with open(label_file, "w", encoding="utf-8") as out:
        for cls, bx1, by1, bx2, by2 in labels:
            ix1 = max(bx1, tx1)
            iy1 = max(by1, ty1)
            ix2 = min(bx2, tx2)
            iy2 = min(by2, ty2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            original_area = max((bx2 - bx1) * (by2 - by1), 1e-9)
            clipped_area = (ix2 - ix1) * (iy2 - iy1)
            if clipped_area / original_area < min_visibility:
                continue

            cx_n = ((ix1 + ix2) / 2 - tx1) / tile_w
            cy_n = ((iy1 + iy2) / 2 - ty1) / tile_h
            bw_n = (ix2 - ix1) / tile_w
            bh_n = (iy2 - iy1) / tile_h

            if 0 < bw_n <= 1 and 0 < bh_n <= 1:
                out.write(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")


def tile_image_and_labels(img_path, label_path, out_img_dir, out_lbl_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping unreadable image: {img_path}")
        return None

    img_h, img_w = img.shape[:2]
    base = Path(img_path).stem

    labels = read_yolo_labels(label_path, img_w, img_h)
    nba = compute_nba(labels, img_w, img_h)
    desired_tiles = compute_desired_tile_count(nba)
    rows, cols = choose_grid_shape(img_w, img_h, desired_tiles)
    grid_boxes = make_grid_boxes(img_w, img_h, rows, cols, grid_overlap)

    for tile_id, tile_box in enumerate(grid_boxes):
        x1, y1, x2, y2 = tile_box
        tile_img = img[y1:y2, x1:x2]

        out_img = os.path.join(out_img_dir, f"{base}_tile{tile_id}.jpg")
        cv2.imwrite(out_img, tile_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        out_lbl = os.path.join(out_lbl_dir, f"{base}_tile{tile_id}.txt")
        write_tile_labels(out_lbl, labels, tile_box)

    return {
        "tile_count": len(grid_boxes),
        "grid": f"{rows}x{cols}",
        "nba": nba,
    }


def write_tiled_data_yaml():
    data_yaml = os.path.join(dataset_path, "data_tiled.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"""path: {dataset_path}
train: train_tiled/images
val: val_tiled/images

names:
  0: swimmer
  1: boat
  2: jetski
  3: life_saving_appliances
  4: buoy
""")
    print(f"\nWrote {data_yaml}")


def main():
    for split in splits:
        img_dir = os.path.join(dataset_path, "images", split)
        lbl_dir = os.path.join(dataset_path, "labels", split)
        out_img_dir = os.path.join(dataset_path, f"{split}_tiled", "images")
        out_lbl_dir = os.path.join(dataset_path, f"{split}_tiled", "labels")

        if not os.path.exists(img_dir):
            print(f"Skipping missing image split: {img_dir}")
            continue
        if not os.path.exists(lbl_dir):
            print(f"Skipping split without labels: {lbl_dir}")
            continue

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        print(f"\nTiling {len(image_files)} {split} images...")

        tile_counts = []
        grids = {}
        for filename in tqdm(image_files):
            img_path = os.path.join(img_dir, filename)
            label_path = os.path.join(lbl_dir, Path(filename).stem + ".txt")
            result = tile_image_and_labels(img_path, label_path, out_img_dir, out_lbl_dir)
            if result:
                tile_counts.append(result["tile_count"])
                grids[result["grid"]] = grids.get(result["grid"], 0) + 1

        if tile_counts:
            print(
                f"   Tiles/image -> min: {min(tile_counts)}, "
                f"max: {max(tile_counts)}, avg: {sum(tile_counts) / len(tile_counts):.2f}"
            )
            print(f"   Grid distribution: {grids}")

    write_tiled_data_yaml()
    print("\nDone. Adaptive tiled dataset created.")


if __name__ == "__main__":
    main()
