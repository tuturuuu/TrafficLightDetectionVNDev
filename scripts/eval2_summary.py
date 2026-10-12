#!/usr/bin/env python
"""Tóm tắt Eval 2: thời gian/epoch và mAP sau mỗi epoch từ runs/eval2/*/results.csv."""
import csv
import glob
import os

rows = []
for path in sorted(glob.glob("runs/eval2/*/results.csv")):
    with open(path) as f:
        r = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    if not r:
        continue
    t = [float(x["time"]) for x in r]                      # giây, cộng dồn
    ep_times = [t[0]] + [b - a for a, b in zip(t, t[1:])]
    rows.append(dict(
        run=os.path.basename(os.path.dirname(path)),
        epochs=len(r),
        min_per_epoch=sum(ep_times[1:] or ep_times) / len(ep_times[1:] or ep_times) / 60,
        first_epoch_min=ep_times[0] / 60,
        map50=[float(x["metrics/mAP50(B)"]) for x in r],
        map50_95=float(r[-1]["metrics/mAP50-95(B)"]),
    ))

print(f"{'run':45s} {'ep':>3s} {'min/ep':>7s} {'ep1 min':>8s} {'mAP50-95@last':>14s}  mAP50 theo epoch")
for x in rows:
    print(f"{x['run']:45s} {x['epochs']:3d} {x['min_per_epoch']:7.1f} {x['first_epoch_min']:8.1f} "
          f"{x['map50_95']:14.4f}  " + " ".join(f"{m:.3f}" for m in x["map50"]))
print("\nmin/ep = trung bình các epoch sau epoch 1 (epoch 1 có overhead cache/khởi tạo).")
print("Lưu ý: val của Kayuan đang bị rò rỉ từ train (split theo frame) -> chỉ so TƯƠNG ĐỐI pretrained vs scratch.")
