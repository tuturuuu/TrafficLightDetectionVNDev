## Step 1 — Train the grid net
Uses **full images + YOLO labels directly** — no pre-tiled dataset needed.

```bash
python scripts/train_selector.py configs/train/selector_kayuan.yaml
# đổi tham số: gridnet.pos_weight=8 gridnet.epochs=60 out=weights/gridnet_pw8.pth
```

Watch two numbers per epoch:
- **cell-recall** — fraction of object-containing grid cells kept.
  This must stay very high (≥0.98); every missed cell is a guaranteed
  missed detection. Raise `gridnet.pos_weight` (try 5–10) if it's low.
- **keep-frac** — fraction of cells kept overall. This is your tile
  reduction: keep-frac 0.35 ≈ 65% of tiles skipped.

The tension between these two IS the method. For the paper you will sweep
`tile_selector.gridnet.threshold` to trace the recall/efficiency trade-off.

## Step 2 — Sanity-check the selection overhead

```bash
python scripts/benchmark_gridnet.py benchmark
```


```bash
python scripts/benchmark_gridnet.py benchmark \
    --grid-rows 8 --grid-cols 8 \
    --images /home/vietpham/dataset/dataset/test/images \
    --labels /home/vietpham/dataset/dataset/test/labels \
    --threshold 0.2 \
    --benchmark-out outputs/benchmark_results \
    --model weights/grid_net.pth
```

Expect ~1–3 ms/image on GPU. Over 1000 images that is 1–3 s of total
overhead — the budget the tile savings must beat. (Compare: your per-tile
CNN pipeline added ~13 s.)

## Step 3 — Run the scaling sweep

```bash
python scripts/evaluate.py configs/eval/kayuan.yaml
# đổi tham số: tiling.tile_size=[640,320] tile_selector.gridnet.threshold=0.2
```

(The old `legacy/dense_tiling_experiment.py` does the same sweep but its
accuracy is NOT class-aware; `scripts/evaluate.py` reproduces its predictions
exactly — see `test/test_parity.py` — and reports class-aware mAP50/mAP50-95.)

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

## Timing fairness rules (enforced by `adaptile.pipeline.run_dataset` — cite in the paper)

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
- **Adaptive wins but recall drops >1 pp** → sweep `tile_selector.gridnet.threshold` and report
  the Pareto front instead of a single point.

## Caveat on accuracy at small tile sizes

As tiles shrink, uniform-tiling mAP itself may change (more boundary
objects, more upscaling). That's fine — the comparison is adaptive vs
uniform *at the same tile size*, column-by-column. Don't compare mAP across
tile sizes as if it were the same task.
