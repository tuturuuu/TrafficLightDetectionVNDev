"""
Cắt dataset thành tile uniform (offline) để train detector.

`tile_image_and_labels` chuyển nguyên văn từ tiling/tiling.py cũ (nay là scripts/tile_dataset.py). `ensure_tiled`
cắt các split còn thiếu và ghi `tiling.json` vào thư mục output để lần sau
dùng lại mà không cắt lại, đồng thời báo lỗi nếu tham số khác.
"""

import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm


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


IMAGE_EXTS = (".jpg", ".png", ".jpeg")


def tile_split(img_dir, lbl_dir, out_dir, tile_size, overlap, min_visibility):
    """Cắt mọi ảnh trong img_dir -> out_dir/{images,labels} + out_dir/tiling.json."""
    out_img, out_lbl = Path(out_dir) / "images", Path(out_dir) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS))
    for f in tqdm(files, desc=f"tiling {Path(img_dir).parent.name}/{Path(img_dir).name}"):
        tile_image_and_labels(
            os.path.join(img_dir, f),
            os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"),
            str(out_img), str(out_lbl), tile_size, overlap, min_visibility)
    params = {"tile_size": tile_size, "overlap": overlap,
              "min_visibility": min_visibility, "source": str(img_dir),
              "n_images": len(files)}
    with open(Path(out_dir) / "tiling.json", "w") as fh:
        json.dump(params, fh, indent=2)


def ensure_tiled(img_dir, lbl_dir, out_dir, tile_size, overlap, min_visibility):
    """Dùng lại out_dir nếu đã cắt với cùng tham số; chưa có thì cắt.

    Thư mục tile cũ (không có tiling.json) được dùng lại kèm cảnh báo, vì không
    biết chắc tham số đã dùng."""
    out_dir = Path(out_dir)
    marker = out_dir / "tiling.json"
    want = {"tile_size": tile_size, "overlap": overlap, "min_visibility": min_visibility}
    if marker.exists():
        have = json.loads(marker.read_text())
        diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
        if diff:
            raise RuntimeError(
                f"{out_dir} đã được cắt với tham số khác {diff} (cũ, mới). "
                "Đổi tiling.suffix hoặc xóa thư mục đó.")
        return "reused"
    if (out_dir / "images").is_dir() and any((out_dir / "images").iterdir()):
        print(f"⚠️  {out_dir} đã có tile nhưng không có tiling.json; dùng lại, "
              f"KHÔNG kiểm tra được tham số (mong đợi {want}).")
        return "reused-unverified"
    tile_split(img_dir, lbl_dir, out_dir, tile_size, overlap, min_visibility)
    return "created"
