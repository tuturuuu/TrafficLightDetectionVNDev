## Step 1 — Train the grid net
Uses **full images + YOLO labels directly** — no pre-tiled dataset needed.

```bash
python grid_proposal_net.py train \
    --images /home/vietpham/dataset/dataset/train/images \
    --labels /home/vietpham/dataset/dataset/train/labels \
    --grid-rows 8 --grid-cols 8 \
    --epochs 40 --out grid_net.pth
```

Watch two numbers per epoch:
- **cell-recall** — fraction of object-containing grid cells kept.
  This must stay very high (≥0.98); every missed cell is a guaranteed
  missed detection. Raise `--pos-weight` (try 5–10) if it's low.
- **keep-frac** — fraction of cells kept overall. This is your tile
  reduction: keep-frac 0.35 ≈ 65% of tiles skipped.

The tension between these two IS the method. For the paper you will sweep
`--threshold` to trace the recall/efficiency trade-off.

## Step 2 — Sanity-check the selection overhead

```bash
python grid_proposal_net.py benchmark
```


```bash
python grid_proposal_net.py benchmark \
    --grid-rows 8 --grid-cols 8 \
    --images /home/vietpham/dataset/dataset/test/images \
    --labels /home/vietpham/dataset/dataset/test/labels \
    --threshold 0.2 \
    --benchmark-out benchmark_results \
    --model grid_net.pth
```

Expect ~1–3 ms/image on GPU. Over 1000 images that is 1–3 s of total
overhead — the budget the tile savings must beat. (Compare: your per-tile
CNN pipeline added ~13 s.)

## Step 3 — Run the scaling sweep

```bash
python dense_tiling_experiment.py \
    --images /home/vietpham/dataset/dataset/test/images \
    --labels /home/vietpham/dataset/dataset/test/labels \
    --model  /home/vietpham/projects/yolo11m_for_label1/runs/detect/new/yolo26_traffic_light_dataset2_tiling3/weights/best.pt  \
    --gridnet grid_net.pth \
    --tile-sizes 640 480 320 240 160 \
    --threshold 0.2 \
    --out scaling_results.json
```

On ~1920×1080 images with 20% overlap this gives roughly:

| tile size | tiles/img |
|---|---|
| 640 | ~6 (your current regime) |
| 480 | ~12 |
| 320 | ~28 |
| 240 | ~54 |
| 160 | ~117 |

The script prints a break-even table and marks the rows where adaptive wins
(faster AND within 1 pp mAP). That table goes straight into the paper.

## Timing fairness rules (already enforced by the script — cite in the paper)

1. `imread` done once, shared, excluded from both methods
2. Identical crop / letterbox / NMS-merge code for both methods
3. `torch.cuda.synchronize()` before every timer read
4. 3 warmup images excluded
5. The ONLY differences timed: one grid-net pass vs zero, and fewer YOLO calls

## What each outcome means for the paper

- **Adaptive wins at ≥~25 tiles/img, loses at 6** → ideal. Contribution:
  "constant-overhead learned tile selection with a characterized break-even
  point." Plot wall-time vs tiles/img for both methods; the crossover is
  your Figure.
- **Adaptive never wins** → the honest conclusion is that YOLO26m per-tile
  cost is too low for selection to matter on this hardware; report it as a
  limitation and keep uniform tiling. Still worth one paragraph.
- **Adaptive wins but recall drops >1 pp** → sweep `--threshold` and report
  the Pareto front instead of a single point.

## Caveat on accuracy at small tile sizes

As tiles shrink, uniform-tiling mAP itself may change (more boundary
objects, more upscaling). That's fine — the comparison is adaptive vs
uniform *at the same tile size*, column-by-column. Don't compare mAP across
tile sizes as if it were the same task.
