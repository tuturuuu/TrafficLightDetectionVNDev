"""
Benchmark GridNet: thời gian chạy mỗi ảnh, cell-recall và ảnh heatmap.

    python scripts/benchmark_gridnet.py benchmark \
        --model weights/gridnet_kayuan_8x8.pth \
        --images data/kayuan/val/images --labels data/kayuan/val/labels \
        --threshold 0.10 --benchmark-out outputs/kayuan/gridnet_val_thr010

Model và phần train nằm ở src/adaptile/models/gridnet.py; train bằng
scripts/train_selector.py. (Trước đây là grid_proposal_net.py.)
"""

import os
import sys
import argparse
import time
from glob import glob

import cv2
import numpy as np

import torch

INPUT_SIZE = 256          # full image downsampled to this square
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from adaptile.models.gridnet import GridProposalNet  # noqa: E402


def benchmark(args):
    if args.model:
        net = GridProposalNet.load(args.model)
    else:
        net = GridProposalNet(args.grid_rows, args.grid_cols).to(DEVICE).eval()
        print("WARNING: no --model given, using random weights (timing only, recall will be meaningless)")

    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # warmup
    for _ in range(10):
        net.propose(img)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        net.propose(img)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"GridProposalNet ({net.count_parameters():,} params, {DEVICE}): "
          f"{ms:.2f} ms/image incl. resize+transfer")
    print(f"Over 1000 images: {ms:.1f} s of total selection overhead")

    # Compute recall and visualize kept tiles if images/labels provided
    if hasattr(args, 'images') and hasattr(args, 'labels') and args.images and args.labels:
        images = sorted(glob(os.path.join(args.images, "*.jpg")) +
                        glob(os.path.join(args.images, "*.png")))
        if images:
            os.makedirs(args.benchmark_out, exist_ok=True)
            tp = fn = 0
            for img_path in images[:100]:  # benchmark on first 100
                stem = os.path.splitext(os.path.basename(img_path))[0]
                img_bgr = cv2.imread(img_path)
                h, w = img_bgr.shape[:2]
                keep, probs = net.propose(img_bgr, threshold=args.threshold)

                # Load ground truth
                lbl = os.path.join(args.labels, stem + ".txt")
                target = np.zeros((args.grid_rows, args.grid_cols), dtype=bool)
                if os.path.exists(lbl):
                    with open(lbl) as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            cx, cy = float(parts[1]), float(parts[2])
                            col = min(int(cx * args.grid_cols), args.grid_cols - 1)
                            row = min(int(cy * args.grid_rows), args.grid_rows - 1)
                            target[row, col] = True

                # Count TP/FN
                tp += (keep & target).sum()
                fn += (~keep & target).sum()

                # Visualize grid overlay
                vis = img_bgr.copy()
                tile_h, tile_w = h // args.grid_rows, w // args.grid_cols
                
                for i in range(args.grid_rows):
                    for j in range(args.grid_cols):
                        y1, y2 = i * tile_h, (i + 1) * tile_h
                        x1, x2 = j * tile_w, (j + 1) * tile_w

                        if keep[i, j]:
                            color = (0, 255, 0)  # Green for kept
                        else:
                            color = (0, 0, 255)  # Red for not kept

                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

                # Save visualization
                out_path = os.path.join(args.benchmark_out, f"{stem}.jpg")
                cv2.imwrite(out_path, vis)

            recall = tp / max(tp + fn, 1)
            print(f"Recall (first 100 images): {recall:.3f}")
            print(f"Saved visualizations to {args.benchmark_out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("benchmark")
    pb.add_argument("--grid-rows", type=int, default=8)
    pb.add_argument("--grid-cols", type=int, default=8)
    pb.add_argument("--images")
    pb.add_argument("--labels")
    pb.add_argument("--threshold", type=float, default=0.2)
    pb.add_argument("--benchmark-out", default="outputs/benchmark_results")
    pb.add_argument("--model", help="path to trained checkpoint (e.g. weights/gridnet_kayuan_8x8.pth)")


    args = p.parse_args()
    if args.cmd == "benchmark":
        benchmark(args)
