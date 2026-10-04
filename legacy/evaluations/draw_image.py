import os
import re
import cv2
import numpy as np

# ========== CONFIG ==========
TILED_IMG_DIR = "/home/vietpham/projects/yolo11m_for_label1/bstld_yolo_format/train_tiled/images"
TILED_LBL_DIR = "/home/vietpham/projects/yolo11m_for_label1/bstld_yolo_format/train_tiled/labels"
OUTPUT_DIR    = "./merged_visualizations"

# Must match the values used during tiling
ORIG_IMG_DIR  = "/home/vietpham/projects/yolo11m_for_label1/bstld_yolo_format/images/train"
TILE_SIZE     = 740
OVERLAP       = 0.2
# ============================

COLORS = [
    (55, 138, 221), (29, 158, 117), (216, 90, 48), (212, 83, 126),
    (186, 117, 23), (226, 75, 74), (127, 119, 221), (59, 109, 17),
]

def get_color(cls_id):
    return COLORS[cls_id % len(COLORS)]

def compute_grid(orig_h, orig_w, tile_size, overlap):
    """Replicates get_starts() from the tiling script to recover cols and rows."""
    step = int(tile_size * (1 - overlap))

    def get_starts(length):
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, step))
        last_valid = length - tile_size
        if starts[-1] < last_valid:
            starts.append(last_valid)
        return starts

    x_starts = get_starts(orig_w)
    y_starts = get_starts(orig_h)
    return y_starts, x_starts   # rows, cols as start-offset lists

def merge_tiles_and_draw(base_name, img_ext=".jpg", font_scale=0.5, thickness=2):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find the original image to get its dimensions
    orig_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(ORIG_IMG_DIR, base_name + ext)
        if os.path.exists(p):
            orig_path = p
            break
    if orig_path is None:
        raise FileNotFoundError(f"Original image not found for base '{base_name}' in {ORIG_IMG_DIR}")

    orig = cv2.imread(orig_path)
    orig_h, orig_w = orig.shape[:2]

    y_starts, x_starts = compute_grid(orig_h, orig_w, TILE_SIZE, OVERLAP)
    cols = len(x_starts)
    rows = len(y_starts)
    n_tiles = cols * rows
    print(f"{base_name}: original {orig_w}x{orig_h} → grid {cols}×{rows} ({n_tiles} tiles)")

    # Load tiles and labels in tile_id order (same loop order as tiling script: y outer, x inner)
    tiles, labels = [], []
    tile_id = 0
    for ty in y_starts:
        for tx in x_starts:
            img_path = os.path.join(TILED_IMG_DIR, f"{base_name}_tile{tile_id}{img_ext}")
            lbl_path = os.path.join(TILED_LBL_DIR, f"{base_name}_tile{tile_id}.txt")

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Tile not found: {img_path}")

            tiles.append((cv2.imread(img_path), tx, ty))

            tile_labels = []
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:])
                            tile_labels.append((cls, cx, cy, bw, bh))
            labels.append(tile_labels)
            tile_id += 1

    # Build merged canvas at original resolution
    merged = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for (tile_img, tx, ty), _ in zip(tiles, labels):
        th = min(TILE_SIZE, orig_h - ty)
        tw = min(TILE_SIZE, orig_w - tx)
        merged[ty:ty+th, tx:tx+tw] = tile_img[:th, :tw]

    # Draw bounding boxes using original pixel coords
    for (tile_img, tx, ty), tile_labels in zip(tiles, labels):
        for cls, cx, cy, bw, bh in tile_labels:
            # Tile-local YOLO → absolute pixel in merged image
            x1 = int((cx - bw / 2) * TILE_SIZE + tx)
            y1 = int((cy - bh / 2) * TILE_SIZE + ty)
            x2 = int((cx + bw / 2) * TILE_SIZE + tx)
            y2 = int((cy + bh / 2) * TILE_SIZE + ty)

            color = get_color(cls)
            cv2.rectangle(merged, (x1, y1), (x2, y2), color, thickness)

            label_text = str(cls)
            (lw, lh), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(merged, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), color, -1)
            cv2.putText(merged, label_text, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_merged_viz.jpg")
    cv2.imwrite(out_path, merged)
    total_boxes = sum(len(l) for l in labels)
    print(f"Saved: {out_path}  ({orig_w}x{orig_h}px, {total_boxes} boxes)")


if __name__ == "__main__":
    # Single image
    merge_tiles_and_draw("9490")

    # --- Batch: all base names in tiled image dir ---
    # base_names = set()
    # for f in os.listdir(TILED_IMG_DIR):
    #     m = re.match(r"^(.+)_tile\d+\.(jpg|jpeg|png)$", f, re.IGNORECASE)
    #     if m:
    #         base_names.add(m.group(1))
    # for base in sorted(base_names):
    #     merge_tiles_and_draw(base)