"""
Dense Tiling Scaling Experiment
================================
Answers the paper's key question:

    "At what tile density does adaptive (grid-net-guided) tiling
     start to beat exhaustive uniform tiling on wall-clock time,
     and what accuracy does it retain?"

For each tile size in the sweep (e.g. 640, 480, 320, 240, 160):
    1. UNIFORM  : cut every tile → YOLO on all tiles → merge
    2. ADAPTIVE : one GridProposalNet pass → cut & YOLO only kept tiles → merge

Both pipelines share IDENTICAL crop / letterbox / merge code, so the ONLY
differences are (a) one grid-net forward per image and (b) fewer YOLO calls.
That is the fair comparison a reviewer will demand.

Timing protocol (important for the paper):
    - torch.cuda.synchronize() before each timer read
    - 3 warmup images excluded from timing
    - Wall time = everything: imread once (shared, excluded), crop,
      grid-net, YOLO, merge
    - Report mean over the test set + tiles/image

Output: results table (printed + JSON) with one row per (tile_size, method),
plus a break-even summary.

Usage:
    python dense_tiling_experiment.py \
        --images  /path/test/images \
        --labels  /path/test/labels \
        --model   yolo26m_finetuned.pt \
        --gridnet grid_net.pth \
        --tile-sizes 640 480 320 240 160 \
        --threshold 0.2 \
        --out scaling_results.json
"""

import os
import json
import time
import argparse
from glob import glob

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared tiling geometry — identical for both methods
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(H, W, tile_size, overlap=0.2):
    """Return list of (x1,y1,x2,y2, row, col) tiles covering the image."""
    stride = max(1, int(tile_size * (1 - overlap)))
    xs = list(range(0, max(W - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - tile_size, 0) + 1, stride)) or [0]
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    xs = sorted(set(max(0, x) for x in xs))
    ys = sorted(set(max(0, y) for y in ys))
    tiles = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            tiles.append((x, y, min(x + tile_size, W), min(y + tile_size, H),
                          r, c))
    return tiles, len(ys), len(xs)


# def grid_mask_for_tiles(keep_mask, n_rows, n_cols, gr, gc):
#     """
#     Map the grid-net's (gr x gc) keep mask onto the actual (n_rows x n_cols)
#     tile grid via nearest-cell lookup, so one trained net serves all tile sizes.
#     """
#     out = np.zeros((n_rows, n_cols), dtype=bool)
#     for r in range(n_rows):
#         for c in range(n_cols):
#             gr_i = min(int((r + 0.5) / n_rows * gr), gr - 1)
#             gc_i = min(int((c + 0.5) / n_cols * gc), gc - 1)
#             out[r, c] = keep_mask[gr_i, gc_i]
#     return out

def grid_mask_for_tiles(keep_mask, tiles, H, W, gr, gc):
    """OR-aggregate net cell decisions over each tile's real footprint."""
    out = np.zeros(len(tiles), dtype=bool)
    cell_h, cell_w = H / gr, W / gc
    for idx, (x1, y1, x2, y2, r, c) in enumerate(tiles):
        # net cell index range that this tile's pixel box overlaps
        r0 = int(y1 / cell_h); r1 = min(int((y2 - 1) / cell_h), gr - 1)
        c0 = int(x1 / cell_w); c1 = min(int((x2 - 1) / cell_w), gc - 1)
        out[idx] = keep_mask[r0:r1+1, c0:c1+1].any()
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Detection + NMS (shared)
# ─────────────────────────────────────────────────────────────────────────────

def yolo_on_tiles(model, image, tiles, conf, imgsz, batch=16):
    """Run YOLO on a list of tiles (batched). Returns list of dets in
    full-image coords: (x1,y1,x2,y2,score,cls)."""
    dets = []
    crops, offsets = [], []
    for (x1, y1, x2, y2, _, _) in tiles:
        crops.append(image[y1:y2, x1:x2])
        offsets.append((x1, y1))
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        offs = offsets[i:i + batch]
        results = model.predict(chunk, conf=conf, imgsz=imgsz, verbose=False)
        for res, (ox, oy) in zip(results, offs):
            if res.boxes is None:
                continue
            b = res.boxes
            xyxy = b.xyxy.cpu().numpy()
            scores = b.conf.cpu().numpy()
            clss = b.cls.cpu().numpy()
            for (x1, y1, x2, y2), s, c in zip(xyxy, scores, clss):
                dets.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                             float(s), int(c)))
    return dets


def nms_merge(dets, iou_thr=0.5):
    if not dets:
        return []
    out = []
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d[5], []).append(d)
    for cls, ds in by_cls.items():
        ds = sorted(ds, key=lambda d: d[4], reverse=True)
        while ds:
            best = ds.pop(0)
            out.append(best)
            ds = [d for d in ds if _iou(best, d) < iou_thr]
    return out


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy (mAP@0.5 + P/R via greedy matching)
# ─────────────────────────────────────────────────────────────────────────────

def load_gt(label_path, H, W):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                              (cx + bw / 2) * W, (cy + bh / 2) * H))
    return boxes


def evaluate_accuracy(all_preds, all_gts, iou_thr=0.5):
    """all_preds: list per image of dets; all_gts: list per image of gt boxes."""
    records = []      # (conf, is_tp)
    n_gt = sum(len(g) for g in all_gts)
    for preds, gts in zip(all_preds, all_gts):
        matched = [False] * len(gts)
        for d in sorted(preds, key=lambda d: d[4], reverse=True):
            best, bi = 0.0, -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                i = _iou(d, (*g, 0, 0))
                if i > best:
                    best, bi = i, j
            if best >= iou_thr and bi >= 0:
                matched[bi] = True
                records.append((d[4], True))
            else:
                records.append((d[4], False))
    if not records or n_gt == 0:
        return 0.0, 0.0, 0.0
    records.sort(key=lambda r: r[0], reverse=True)
    tps = np.cumsum([r[1] for r in records])
    fps = np.cumsum([not r[1] for r in records])
    prec = tps / (tps + fps)
    rec = tps / n_gt
    # 101-point AP
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t]
        ap += (p.max() if len(p) else 0.0)
    ap /= 101
    return float(ap), float(prec[-1]), float(rec[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    from ultralytics import YOLO
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from grid_proposal_net import GridProposalNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    gridnet = GridProposalNet.load(args.gridnet, device=device)
    gr, gc = gridnet.grid_rows, gridnet.grid_cols
    print(f"GridNet {gridnet.count_parameters():,} params, "
          f"grid {gr}x{gc}, device {device}")

    image_paths = sorted(glob(os.path.join(args.images, "*.jpg")) +
                         glob(os.path.join(args.images, "*.png")))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"{len(image_paths)} test images")

    # Preload images once — imread excluded from both methods' timing
    images, gts = [], []
    for p in image_paths:
        img = cv2.imread(p)
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(p))[0]
        images.append(img)
        gts.append(load_gt(os.path.join(args.labels, stem + ".txt"), H, W))

    results = []
    for tile_size in args.tile_sizes:
        for method in ("uniform", "adaptive"):
            preds_all = []
            total_tiles = 0
            kept_tiles = 0
            t_select = 0.0
            t_detect = 0.0

            # warmup (3 images, untimed)
            for img in images[:3]:
                tiles, nr, nc = make_grid(*img.shape[:2], tile_size,
                                          args.overlap)
                if method == "adaptive":
                    gridnet.propose(img, threshold=args.threshold)
                yolo_on_tiles(model, img, tiles[:2], args.conf, args.imgsz)
            if device == "cuda":
                torch.cuda.synchronize()

            t_wall0 = time.perf_counter()
            for img in images:
                H, W = img.shape[:2]
                tiles, nr, nc = make_grid(H, W, tile_size, args.overlap)
                total_tiles += len(tiles)

                if method == "adaptive":
                    t0 = time.perf_counter()
                    keep_coarse, _ = gridnet.propose(
                        img, threshold=args.threshold)
                    keep = grid_mask_for_tiles(keep_coarse, tiles, H, W, gr, gc)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t_select += time.perf_counter() - t0
                    tiles_run = [t for t, k in zip(tiles, keep) if k]
                    if not tiles_run:                     # safety floor
                        tiles_run = tiles[:1]
                else:
                    tiles_run = tiles
                kept_tiles += len(tiles_run)

                t0 = time.perf_counter()
                dets = yolo_on_tiles(model, img, tiles_run,
                                     args.conf, args.imgsz)
                if device == "cuda":
                    torch.cuda.synchronize()
                t_detect += time.perf_counter() - t0

                preds_all.append(nms_merge(dets, args.merge_iou))

            if device == "cuda":
                torch.cuda.synchronize()
            wall = time.perf_counter() - t_wall0

            ap50, prec, rec = evaluate_accuracy(preds_all, gts)
            row = {
                "tile_size": tile_size,
                "method": method,
                "tiles_per_img": round(total_tiles / len(images), 2),
                "kept_per_img": round(kept_tiles / len(images), 2),
                "tile_reduction": round(1 - kept_tiles / max(total_tiles, 1), 4),
                "select_s": round(t_select, 2),
                "detect_s": round(t_detect, 2),
                "wall_s": round(wall, 2),
                "mAP50": round(ap50, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
            results.append(row)
            print(f"tile={tile_size:4d} {method:8s} "
                  f"tiles/img={row['tiles_per_img']:6.1f} "
                  f"kept={row['kept_per_img']:6.1f} "
                  f"wall={row['wall_s']:7.1f}s "
                  f"mAP50={row['mAP50']:.4f} R={row['recall']:.4f}")

    # ── Break-even summary ────────────────────────────────────────────────
    print("\n=== BREAK-EVEN SUMMARY (adaptive vs uniform) ===")
    print(f"{'tile':>5} {'tiles/img':>9} {'Δwall':>8} {'speedup':>8} "
          f"{'ΔmAP50':>8} {'Δrecall':>8}")
    by_size = {}
    for r in results:
        by_size.setdefault(r["tile_size"], {})[r["method"]] = r
    for ts, d in sorted(by_size.items(), reverse=True):
        u, a = d["uniform"], d["adaptive"]
        dw = a["wall_s"] - u["wall_s"]
        sp = u["wall_s"] / max(a["wall_s"], 1e-9)
        dm = (a["mAP50"] - u["mAP50"]) * 100
        dr = (a["recall"] - u["recall"]) * 100
        marker = "  ← adaptive wins" if dw < 0 and dm > -1.0 else ""
        print(f"{ts:>5} {u['tiles_per_img']:>9.1f} {dw:>+7.1f}s "
              f"{sp:>7.2f}x {dm:>+7.2f}pp {dr:>+7.2f}pp{marker}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--gridnet", required=True)
    p.add_argument("--tile-sizes", type=int, nargs="+",
                   default=[640, 480, 320, 240, 160])
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--merge-iou", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", default="scaling_results.json")
    run(p.parse_args())
