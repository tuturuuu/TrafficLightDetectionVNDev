#!/usr/bin/env python
"""
Đánh giá pipeline Tiler -> TileSelector -> Detector -> Merger theo config.

Toàn bộ thí nghiệm nằm trong MỘT file config (xem configs/eval/kayuan.yaml).
Với mỗi `tiling.tile_size` và mỗi `tile_selector.type`, chạy cả tập ảnh (đọc
sẵn, warmup, đo giờ như RUNBOOK.md), rồi tính các metric trong `eval.metrics`.
Thay cho legacy/dense_tiling_experiment.py và legacy/evaluations/evaluation_*.py.

    python scripts/evaluate.py configs/eval/kayuan.yaml
    python scripts/evaluate.py configs/eval/kayuan.yaml tiling.tile_size=[640,320] tile_selector.gridnet.threshold=0.05
    python scripts/evaluate.py configs/eval/kayuan.yaml tile_selector.type=[none,tile_cnn,gridnet]
    python scripts/evaluate.py configs/eval/kayuan.yaml eval.max_images=50 eval.out=/tmp/x.json

eval: map50, map50_95, precision, recall (đều XÉT CLASS), agnostic (số theo
cách cũ của legacy/dense_tiling_experiment.py, không xét class), fps (ảnh / wall time).

Output: JSON danh sách row (các key giống JSON cũ của legacy/dense_tiling_experiment.py,
cộng các metric trong eval), kèm file .yaml ghi lại config đã resolve.
"""

import argparse
import json
import os
import sys

import cv2
from omegaconf import OmegaConf

from adaptile import DETECTORS, MERGERS, SELECTORS, TILERS, Pipeline, evaluate, run_dataset
from adaptile.config import load_config
from adaptile.data import label_path_for, list_images, load_gt

# tên trong `eval` -> key trong kết quả của adaptile.evaluate / trong JSON
ACCURACY_KEYS = {"map50": "mAP50", "map50_95": "mAP50-95",
                 "precision": "precision", "recall": "recall"}


def load_dataset(cfg):
    paths = list_images(cfg.dataset.images, cfg.max_images)
    if not paths:
        sys.exit(f"Không có ảnh trong {cfg.dataset.images}")
    images, gts = [], []
    for p in paths:
        img = cv2.imread(p)
        images.append(img)
        gts.append(load_gt(label_path_for(p, cfg.dataset.labels), *img.shape[:2]))
    return images, gts


def make_row(tile_size, sel_name, run, n_img, gts, metrics):
    row = {
        "tile_size": tile_size,
        "method": ("uniform" if tile_size else "notile") if sel_name == "none" else sel_name,
        "tiles_per_img": round(run.total_tiles / n_img, 2),
        "kept_per_img": round(run.kept_tiles / n_img, 2),
        "tile_reduction": round(1 - run.kept_tiles / max(run.total_tiles, 1), 4),
        "select_s": round(run.select_s, 2),
        "detect_s": round(run.detect_s, 2),
        "wall_s": round(run.wall_s, 2),
    }
    if "fps" in metrics:
        row["fps"] = round(n_img / max(run.wall_s, 1e-9), 2)
    if set(metrics) & set(ACCURACY_KEYS):
        m = evaluate(run.preds, gts, class_aware=True)
        row.update({key: round(m[key], 4) for name, key in ACCURACY_KEYS.items()
                    if name in metrics})
    if "agnostic" in metrics:
        m = evaluate(run.preds, gts, class_aware=False)
        row["agnostic"] = {k: round(m[k], 4) for k in ("mAP50", "precision", "recall")}
    return row


def print_summary(rows):
    by_size = {}
    for r in rows:
        by_size.setdefault(r["tile_size"], {})[r["method"]] = r
    methods = {r["method"] for r in rows}
    others = sorted(methods - {"uniform"})
    if "uniform" not in methods or not others:
        return
    print("\n=== SO VỚI UNIFORM ===")
    deltas = [k for k in ("mAP50", "recall") if k in rows[0]]
    print(f"{'tile':>5} {'method':>9} {'kept/img':>9} {'speedup':>8} "
          + " ".join(f"{'Δ' + k:>8}" for k in deltas))
    for ts, d in by_size.items():
        u = d["uniform"]
        for name in others:
            a = d[name]
            sp = u["wall_s"] / max(a["wall_s"], 1e-9)
            print(f"{str(ts):>5} {name:>9} {a['kept_per_img']:>9.1f} {sp:>7.2f}x "
                  + " ".join(f"{(a[k] - u[k]) * 100:>+7.2f}pp" for k in deltas))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="file config, vd configs/eval/kayuan.yaml")
    p.add_argument("overrides", nargs="*",
                   help="key=value, vd tile_selector.gridnet.threshold=0.05 eval.max_images=50")
    p.add_argument("--overwrite", action="store_true", help="cho phép ghi đè file out")
    args = p.parse_intermixed_args()   # cho phép đặt --cờ ở bất kỳ đâu

    cfg = load_config(args.config, args.overrides)
    print(OmegaConf.to_yaml(cfg.source))
    if os.path.exists(cfg.out) and not args.overwrite:
        sys.exit(f"{cfg.out} đã tồn tại; đổi eval.out=... hoặc thêm --overwrite")

    images, gts = load_dataset(cfg)
    print(f"{len(images)} ảnh ({cfg.dataset.name}: {cfg.dataset.images})")

    detector = DETECTORS.build(cfg.detector)
    merger = MERGERS.build(cfg.merger)
    selectors = {name: SELECTORS.build(c) for name, c in cfg.selectors.items()}
    tile_sizes = cfg.tile_size if cfg.tiler.name != "none" else [None]

    rows = []
    for ts in tile_sizes:
        tiler_cfg = OmegaConf.to_container(cfg.tiler)
        if ts is not None:
            tiler_cfg["tile_size"] = ts
        tiler = TILERS.build(tiler_cfg)
        for name, sel in selectors.items():
            run = run_dataset(Pipeline(tiler, sel, detector, merger), images, cfg.n_warmup)
            row = make_row(ts, name, run, len(images), gts, cfg.metrics)
            rows.append(row)
            shown = [k for k in ("fps", *ACCURACY_KEYS.values()) if k in row]
            print(f"tile={str(ts):>4} {row['method']:8s} "
                  f"tiles/img={row['tiles_per_img']:6.1f} kept={row['kept_per_img']:6.1f} "
                  f"wall={row['wall_s']:7.1f}s "
                  + " ".join(f"{k}={row[k]:.1f}" if k == "fps" else f"{k}={row[k]:.4f}"
                             for k in shown), flush=True)

    print_summary(rows)
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(rows, f, indent=2)
    OmegaConf.save(cfg.source, os.path.splitext(cfg.out)[0] + ".yaml")
    print(f"\nĐã lưu {cfg.out}")


if __name__ == "__main__":
    main()
