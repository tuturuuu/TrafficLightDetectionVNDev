import os
import cv2
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ========== CONFIG ==========
dataset_path = "/home/vietpham/dataset/dataset"
splits = ["train", "val", "test"]    # process all 3 splits
tile_size = 640                      # tile dimension (640x640)
target_nba = 0.04                    # target Normalized Bounding Box Area
min_tiles = 1                        # minimum number of tiles per axis
max_tiles = 6                        # cap to avoid explosion in tile count
overlap_factor = 1.5                 # overlap = 1.5x average object size (as per paper)
fallback_overlap = 0.2               # fallback overlap ratio if no labels exist
# ============================


def compute_nba(labels, img_w, img_h):
    """
    Compute the median Normalized Bounding Box Area (NBA) from YOLO labels.
    NBA = bounding_box_area / image_area
    Uses the 10th percentile to account for the smallest objects (worst case).
    """
    if not labels:
        return None
    img_area = img_w * img_h
    nbas = []
    for _, x1, y1, x2, y2 in labels:
        box_area = (x2 - x1) * (y2 - y1)
        nbas.append(box_area / img_area)
    # Use 10th percentile so small objects drive the tiling decision
    return float(np.percentile(nbas, 10))


def compute_num_tiles(nba, target_nba, min_tiles, max_tiles):
    """
    Compute how many tiles per axis are needed to reach target_nba.

    Since splitting into N×N tiles makes each tile cover 1/N² of the image,
    objects appear N² times larger in NBA terms.

    So: current_nba * N² >= target_nba
        N >= sqrt(target_nba / current_nba)
    """
    if nba is None or nba <= 0:
        return min_tiles
    n = math.ceil(math.sqrt(target_nba / nba))
    return max(min_tiles, min(n, max_tiles))


def compute_overlap_pixels(labels, img_w, img_h):
    """
    Compute overlap in pixels as 1.5x the average object width/height.
    Falls back to fallback_overlap ratio if no labels.
    """
    if not labels:
        return None  # signal to use fallback
    widths = [(x2 - x1) for _, x1, y1, x2, y2 in labels]
    heights = [(y2 - y1) for _, x1, y1, x2, y2 in labels]
    avg_size = np.mean(widths + heights)
    return int(overlap_factor * avg_size)


def tile_image_and_labels(img_path, label_path, out_img_dir, out_lbl_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping unreadable image: {img_path}")
        return
    h, w = img.shape[:2]
    base = Path(img_path).stem

    # ── Read YOLO labels ──────────────────────────────────────────────────────
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
                x1 = cx_px - bw_px / 2
                y1 = cy_px - bh_px / 2
                x2 = cx_px + bw_px / 2
                y2 = cy_px + bh_px / 2
                labels.append((int(cls), x1, y1, x2, y2))

    # ── Adaptive tiling decision ──────────────────────────────────────────────
    nba = compute_nba(labels, w, h)
    n_tiles = compute_num_tiles(nba, target_nba, min_tiles, max_tiles)

    # Overlap: either label-driven (1.5× avg object size) or fallback ratio
    overlap_px = compute_overlap_pixels(labels, w, h)
    if overlap_px is None:
        # No labels: fall back to a fixed ratio of the tile size
        overlap_px = int(tile_size * fallback_overlap)

    if n_tiles == 1:
        # No tiling needed — image is already close enough to target NBA
        tile_w = w
        tile_h = h
        step_x = w
        step_y = h
    else:
        tile_w = tile_size
        tile_h = tile_size
        # Step = tile_size minus the label-driven overlap
        step_x = max(1, tile_size - overlap_px)
        step_y = max(1, tile_size - overlap_px)

    # ── Tile and write outputs ────────────────────────────────────────────────
    tile_id = 0
    for ty in range(0, max(1, h - tile_h + 1), step_y):
        for tx in range(0, max(1, w - tile_w + 1), step_x):
            tx2 = min(w, tx + tile_w)
            ty2 = min(h, ty + tile_h)
            tx1 = tx2 - tile_w
            ty1 = ty2 - tile_h

            tile_img = img[ty1:ty2, tx1:tx2]
            out_img = os.path.join(out_img_dir, f"{base}_tile{tile_id}.jpg")
            cv2.imwrite(out_img, tile_img)

            # Remap labels into tile coordinate space
            out_lbl = os.path.join(out_lbl_dir, f"{base}_tile{tile_id}.txt")
            with open(out_lbl, "w") as lf:
                for cls, bx1, by1, bx2, by2 in labels:
                    ix1 = max(bx1, tx1)
                    iy1 = max(by1, ty1)
                    ix2 = min(bx2, tx2)
                    iy2 = min(by2, ty2)
                    if ix2 > ix1 and iy2 > iy1:
                        cx_n = ((ix1 + ix2) / 2 - tx1) / tile_w
                        cy_n = ((iy1 + iy2) / 2 - ty1) / tile_h
                        bw_n = (ix2 - ix1) / tile_w
                        bh_n = (iy2 - iy1) / tile_h
                        if 0 < bw_n <= 1 and 0 < bh_n <= 1:
                            lf.write(f"{cls} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")
            tile_id += 1

    return n_tiles, nba  # useful for logging/debugging


# ========== RUN TILING ==========
for split in splits:
    img_dir = os.path.join(dataset_path, f"{split}/images/")
    lbl_dir = os.path.join(dataset_path, f"{split}/labels/")
    out_img_dir = os.path.join(dataset_path, f"{split}_tiled/images")
    out_lbl_dir = os.path.join(dataset_path, f"{split}_tiled/labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"\n📸 Tiling {len(image_files)} {split} images...")

    tile_counts = []
    for f in tqdm(image_files):
        img_path = os.path.join(img_dir, f)
        lbl_path = os.path.join(lbl_dir, Path(f).stem + ".txt")
        result = tile_image_and_labels(img_path, lbl_path, out_img_dir, out_lbl_dir)
        if result:
            n_tiles, nba = result
            tile_counts.append(n_tiles)

    if tile_counts:
        print(f"   Tiles/axis → min: {min(tile_counts)}, max: {max(tile_counts)}, "
              f"avg: {sum(tile_counts)/len(tile_counts):.1f}")

print("\n✅ Done! Adaptive tiled dataset created.")