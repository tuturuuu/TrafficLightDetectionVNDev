#!/usr/bin/env python3
"""Collect experiment metrics for SeaDroneSees report tables.

Reads Ultralytics training `results.csv` files and, optionally, runs
validation on each `best.pt` checkpoint to collect latency/FPS.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import re
from pathlib import Path


DEFAULT_RUNS_GLOB = "runs/detect/runs/seadronesees_yolov8_cbam_tiled_seed*"
DEFAULT_DATA = "data/SeaDroneSees/data_tiled.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-glob", default=DEFAULT_RUNS_GLOB)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out-csv", default="report_metrics.csv")
    parser.add_argument("--out-md", default="report_metrics.md")
    parser.add_argument("--include-val-speed", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def clean_float(value: str) -> float:
    return float(value.strip())


def infer_seed(run_name: str) -> str:
    match = re.search(r"seed(\d+)$", run_name)
    if not match:
        return ""
    seed_text = match.group(1)
    if seed_text == "02":
        return "0"
    return str(int(seed_text))


def read_training_metrics(run_dir: Path) -> dict:
    results_path = run_dir / "results.csv"
    weights_path = run_dir / "weights" / "best.pt"
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    with results_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {results_path}")

    best = max(rows, key=lambda r: clean_float(r["metrics/mAP50-95(B)"]))
    last = rows[-1]

    return {
        "run": run_dir.name,
        "seed": infer_seed(run_dir.name),
        "best_epoch": int(float(best["epoch"])),
        "precision": clean_float(best["metrics/precision(B)"]),
        "recall": clean_float(best["metrics/recall(B)"]),
        "mAP50": clean_float(best["metrics/mAP50(B)"]),
        "mAP50-95": clean_float(best["metrics/mAP50-95(B)"]),
        "time_min": clean_float(last["time"]) / 60.0,
        "checkpoint": str(weights_path) if weights_path.exists() else "",
        "latency_ms": "",
        "fps": "",
    }


def register_custom_modules():
    import os
    import sys

    import ultralytics.nn.tasks as tasks

    sys.path.append(os.path.abspath("./utils"))
    from custom_modules import CBAM, SE

    tasks.CBAM = CBAM
    tasks.SE = SE


def add_val_speed(row: dict, data: str, device: str, batch: int, imgsz: int) -> None:
    if not row["checkpoint"]:
        return

    register_custom_modules()
    from ultralytics import YOLO

    model = YOLO(row["checkpoint"])
    metrics = model.val(
        data=data,
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
    )

    speed = metrics.speed
    preprocess = float(speed.get("preprocess", 0.0))
    inference = float(speed.get("inference", 0.0))
    postprocess = float(speed.get("postprocess", 0.0))
    latency_ms = preprocess + inference + postprocess

    row["latency_ms"] = latency_ms
    row["fps"] = 1000.0 / latency_ms if latency_ms > 0 else ""

    # Use the fresh validation metrics when available because they are from
    # the saved best.pt checkpoint, not just the best row in training CSV.
    box = metrics.box
    row["precision"] = float(box.mp)
    row["recall"] = float(box.mr)
    row["mAP50"] = float(box.map50)
    row["mAP50-95"] = float(box.map)


def format_value(value, digits=4):
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(rows: list[dict], out_path: Path) -> None:
    fields = [
        "run",
        "seed",
        "best_epoch",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "fps",
        "latency_ms",
        "time_min",
        "checkpoint",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(rows: list[dict], out_path: Path) -> None:
    fields = [
        ("seed", "Seed"),
        ("best_epoch", "Best epoch"),
        ("mAP50", "mAP50"),
        ("mAP50-95", "mAP50-95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("fps", "FPS"),
        ("latency_ms", "Latency (ms)"),
        ("time_min", "Time (min)"),
    ]
    header = "| " + " | ".join(label for _, label in fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(key, "")) for key, _ in fields)
            + " |"
        )
    if rows:
        lines.append("")
        lines.append("Summary:")
        for key, label in [
            ("mAP50", "mAP50"),
            ("mAP50-95", "mAP50-95"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("fps", "FPS"),
            ("latency_ms", "Latency (ms)"),
            ("time_min", "Time (min)"),
        ]:
            values = [
                float(row[key]) for row in rows
                if row.get(key) != "" and row.get(key) is not None
            ]
            if not values:
                continue
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            lines.append(f"- {label}: {mean:.4f} +/- {std:.4f}")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    run_dirs = sorted(
        p for p in Path(".").glob(args.runs_glob)
        if p.is_dir() and (p / "results.csv").exists()
    )
    if not run_dirs:
        raise RuntimeError(f"No run dirs matched: {args.runs_glob}")

    rows = [read_training_metrics(run_dir) for run_dir in run_dirs]
    if args.include_val_speed:
        for row in rows:
            print(f"Validating {row['run']} for speed/FPS...")
            add_val_speed(row, args.data, args.device, args.batch, args.imgsz)

    write_csv(rows, Path(args.out_csv))
    write_markdown(rows, Path(args.out_md))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
