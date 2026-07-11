"""
GridProposalNet — single-pass tile proposal
============================================
Replaces the per-tile TileProposalCNN with ONE forward pass per image.

Key difference from small_cnn.py:
    OLD:  crop 6..100 tiles → preprocess each → CNN on each tile   (N passes)
    NEW:  downsample full image once → CNN → G×G grid of keep-scores (1 pass)

Each output cell corresponds to one tile position in the tiling grid.
Selection overhead is therefore CONSTANT per image, independent of tile count
— exactly the property needed for adaptive tiling to pay off in the
high-tile-count regime.

Training target: cell (i,j) = 1 if any GT box centre falls inside tile (i,j).
Trained directly from full images + YOLO-format labels (no pre-tiled dataset
needed).

Usage (train):
    python grid_proposal_net.py train \
        --images /path/train/images --labels /path/train/labels \
        --grid-rows 8 --grid-cols 8 --epochs 40 --out grid_net.pth

Usage (inference, inside a pipeline):
    net    = GridProposalNet.load("grid_net.pth", grid_rows=8, grid_cols=8)
    keep   = net.propose(image_bgr, threshold=0.2)   # boolean (G_r, G_c) mask
"""

import os
import sys
import argparse
import random
import time
from glob import glob

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

INPUT_SIZE = 256          # full image downsampled to this square
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GridProposalNet(nn.Module):
    """
    Fully-convolutional grid predictor.

    Input : (B, 3, 256, 256) full downsampled image
    Output: (B, 1, grid_rows, grid_cols) keep-logits, one per tile position

    ~75k params — same budget as the per-tile classifier, but runs ONCE
    per image instead of once per tile.
    """

    def __init__(self, grid_rows: int = 8, grid_cols: int = 8):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            block(3, 16),      # 256 -> 128
            block(16, 32),     # 128 -> 64
            block(32, 48),     # 64  -> 32
            block(48, 64),     # 32  -> 16
        )
        # Map 16x16 feature map to grid_rows x grid_cols
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((grid_rows, grid_cols)),
            nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x))     # (B,1,Gr,Gc) logits

    # ── convenience ──────────────────────────────────────────────────────

    @torch.no_grad()
    def propose(self, image_bgr: np.ndarray, threshold: float = 0.2,
                min_keep: int = 1) -> np.ndarray:
        """
        One forward pass → boolean keep-mask of shape (grid_rows, grid_cols).
        Guarantees at least `min_keep` tiles kept (highest-scoring), so an
        overconfident network can never blank out an image entirely.
        """
        self.eval()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
        t = t.unsqueeze(0).to(next(self.parameters()).device)
        probs = torch.sigmoid(self(t))[0, 0].cpu().numpy()   # (Gr,Gc)
        keep = probs > threshold
        if keep.sum() < min_keep:
            flat = probs.flatten()
            top = np.argsort(flat)[::-1][:min_keep]
            keep = np.zeros_like(flat, dtype=bool)
            keep[top] = True
            keep = keep.reshape(probs.shape)
        return keep, probs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):
        torch.save({"state": self.state_dict(),
                    "grid_rows": self.grid_rows,
                    "grid_cols": self.grid_cols}, path)

    @staticmethod
    def load(path, device=DEVICE):
        ckpt = torch.load(path, map_location=device)
        net = GridProposalNet(ckpt["grid_rows"], ckpt["grid_cols"])
        net.load_state_dict(ckpt["state"])
        return net.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: full images + YOLO labels → grid targets
# ─────────────────────────────────────────────────────────────────────────────

class GridDataset(Dataset):
    """
    Builds (image, grid_target) pairs directly from full-resolution images
    and YOLO-format labels. No pre-tiled dataset required.

    Cell (i,j) = 1 if any GT box centre lies in tile (i,j) of an
    (grid_rows x grid_cols) uniform grid over the image.
    """

    def __init__(self, image_paths, label_dir, grid_rows, grid_cols,
                 augment=False):
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.gr, self.gc = grid_rows, grid_cols
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _load_target(self, stem):
        target = np.zeros((self.gr, self.gc), dtype=np.float32)
        lbl = os.path.join(self.label_dir, stem + ".txt")
        if os.path.exists(lbl):
            with open(lbl) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx, cy = float(parts[1]), float(parts[2])   # normalized
                    col = min(int(cx * self.gc), self.gc - 1)
                    row = min(int(cy * self.gr), self.gr - 1)
                    target[row, col] = 1.0
        return target

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(path))[0]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self._load_target(stem)

        if self.augment:
            if random.random() < 0.5:                      # h-flip
                img = img[:, ::-1].copy()
                target = target[:, ::-1].copy()
            if random.random() < 0.5:                      # brightness
                a = random.uniform(0.8, 1.2)
                b = random.randint(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=a, beta=b)

        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0

        return (torch.from_numpy(img),
                torch.from_numpy(target).unsqueeze(0))     # (1,Gr,Gc)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    images = sorted(glob(os.path.join(args.images, "*.jpg")) +
                    glob(os.path.join(args.images, "*.png")))
    if not images:
        raise RuntimeError(f"No images in {args.images}")

    random.seed(0)
    random.shuffle(images)
    n_val = max(1, int(0.1 * len(images)))
    val_imgs, train_imgs = images[:n_val], images[n_val:]

    tr_ds = GridDataset(train_imgs, args.labels, args.grid_rows,
                        args.grid_cols, augment=True)
    va_ds = GridDataset(val_imgs, args.labels, args.grid_rows,
                        args.grid_cols, augment=False)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4)

    net = GridProposalNet(args.grid_rows, args.grid_cols).to(DEVICE)
    print(f"GridProposalNet: {net.count_parameters():,} params  "
          f"grid={args.grid_rows}x{args.grid_cols}  device={DEVICE}")
    print(f"Train {len(tr_ds)} | Val {len(va_ds)}")

    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight]).to(DEVICE))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        tl = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
        tl /= len(tr_ds)
        sched.step()

        net.eval()
        vl = 0.0
        # tile-recall on val: fraction of positive cells kept at threshold
        tp = fn = kept = total = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = net(x)
                vl += crit(logits, y).item() * x.size(0)
                pred = (torch.sigmoid(logits) > args.threshold)
                pos = y > 0.5
                tp += (pred & pos).sum().item()
                fn += (~pred & pos).sum().item()
                kept += pred.sum().item()
                total += pred.numel()
        vl /= len(va_ds)
        cell_recall = tp / max(tp + fn, 1)
        keep_frac = kept / max(total, 1)

        print(f"Epoch {epoch:3d}/{args.epochs}  train {tl:.4f}  val {vl:.4f}  "
              f"cell-recall {cell_recall:.3f}  keep-frac {keep_frac:.3f}")
        print(f"args.pos_weight={args.pos_weight}  args.threshold={args.threshold}")

        if vl < best_val:
            best_val = vl
            patience = 0
            net_cpu_state = {k: v.cpu() for k, v in net.state_dict().items()}
            torch.save({"state": net_cpu_state,
                        "grid_rows": args.grid_rows,
                        "grid_cols": args.grid_cols}, args.out)
        else:
            patience += 1
            if patience >= 10:
                print("Early stopping.")
                break

    print(f"Best model saved to {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# Overhead micro-benchmark
# ─────────────────────────────────────────────────────────────────────────────

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

    pt = sub.add_parser("train")
    pt.add_argument("--images", required=True)
    pt.add_argument("--labels", required=True)
    pt.add_argument("--grid-rows", type=int, default=8)
    pt.add_argument("--grid-cols", type=int, default=8)
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--pos-weight", type=float, default=3.0)
    pt.add_argument("--threshold", type=float, default=0.2)
    pt.add_argument("--out", default="grid_net.pth")

    pb = sub.add_parser("benchmark")
    pb.add_argument("--grid-rows", type=int, default=8)
    pb.add_argument("--grid-cols", type=int, default=8)
    pb.add_argument("--images")
    pb.add_argument("--labels")
    pb.add_argument("--threshold", type=float, default=0.2)
    pb.add_argument("--benchmark-out", default="benchmark_results")
    pb.add_argument("--model", help="path to trained checkpoint (grid_net.pth)")


    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "benchmark":
        benchmark(args)
