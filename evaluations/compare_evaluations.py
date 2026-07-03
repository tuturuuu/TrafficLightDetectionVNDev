#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EVAL_1 = BASE_DIR / "evaluation_with_cnn.py"
EVAL_2 = BASE_DIR / "evaluation_without_cnn.py"


SUMMARY_PATTERNS = {
    "images_processed": re.compile(r"^Images processed:\s*(\d+)"),
    "tiles_generated": re.compile(r"^Tiles generated:\s*(\d+)"),
    "tiles_kept": re.compile(r"^Tiles kept(?: by proposal model)?:\s*(\d+)"),
    "images_with_kept_tiles": re.compile(r"^Images with at least one kept tile:\s*(\d+)"),
    "tile_reduction": re.compile(r"^Tile reduction:\s*([0-9.]+)"),
    "total_time": re.compile(r"^Total time:\s*([0-9.]+)\s*sec"),
    "tile_fps": re.compile(r"^Tile-level FPS:\s*([0-9.]+)"),
    "real_fps": re.compile(r"^REAL image-level FPS \(with tiling\):\s*([0-9.]+)"),
    "precision": re.compile(r"^Precision:\s*([0-9.]+)"),
    "recall": re.compile(r"^Recall:\s*([0-9.]+)"),
    "map50": re.compile(r"^mAP50:\s*([0-9.]+)"),
    "map50_95": re.compile(r"^mAP50-95:\s*([0-9.]+)"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run both evaluation scripts and compare their results")
    parser.add_argument("--dataset-root", default="/home/vietpham/dataset/dataset")
    parser.add_argument("--yolo-model", default=str(BASE_DIR / "../runs/detect/new/yolo26_traffic_light_dataset2_tiling3/weights/best.pt"))
    parser.add_argument("--tile-model", default=str(BASE_DIR / "../cnn_classifier/tile_proposal_cnn_model.pth"))
    parser.add_argument("--tile-size", type=int, default=740)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print raw script output and derived timing diagnostics")
    return parser.parse_args()


def build_common_args(args):
    common = [
        "--dataset-root", args.dataset_root,
        "--yolo-model", args.yolo_model,
        "--tile-size", str(args.tile_size),
        "--overlap", str(args.overlap),
        "--yolo-conf", str(args.yolo_conf),
        "--nms-iou", str(args.nms_iou),
        "--imgsz", str(args.imgsz),
    ]

    if args.device is not None:
        common.extend(["--device", args.device])
    if args.max_images and args.max_images > 0:
        common.extend(["--max-images", str(args.max_images)])
    if args.agnostic_nms:
        common.append("--agnostic-nms")

    return common


def run_script(script_path, extra_args):
    command = [sys.executable, str(script_path), *extra_args]
    start_time = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    wall_time = time.perf_counter() - start_time
    return completed.stdout, completed.stderr, wall_time


def extract_metrics(output):
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        for key, pattern in SUMMARY_PATTERNS.items():
            match = pattern.match(line)
            if match:
                value = match.group(1)
                metrics[key] = int(value) if key in {"images_processed", "tiles_generated", "tiles_kept"} else float(value)
                break

    return metrics


def format_value(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def print_table(rows):
    headers = [
        "Script",
        "Images",
        "Tiles",
        "Kept",
        "Imgs Kept",
        "Reduction",
        "Tiles/Img",
        "Kept/Img",
        "Time (s)",
        "Wall (s)",
        "Tile FPS",
        "Real FPS",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def render_row(row):
        return "| " + " | ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"

    print(render_row(headers))
    print(separator)
    for row in rows:
        print(render_row(row))


def main():
    args = parse_args()
    common_args = build_common_args(args)

    scripts = [
        ("evaluation.py", EVAL_1),
        ("evaluation_2.py", EVAL_2),
    ]

    rows = []
    script_results = []
    for label, script_path in scripts:
        output, stderr, wall_time = run_script(script_path, common_args)
        metrics = extract_metrics(output)

        images_processed = metrics.get("images_processed")
        tiles_generated = metrics.get("tiles_generated")
        tiles_kept = metrics.get("tiles_kept")
        images_with_kept_tiles = metrics.get("images_with_kept_tiles")

        tiles_per_image = (tiles_generated / images_processed) if images_processed else None
        kept_per_image = (tiles_kept / images_processed) if images_processed else None

        rows.append([
            label,
            format_value(images_processed, digits=0),
            format_value(tiles_generated, digits=0),
            format_value(tiles_kept, digits=0),
            format_value(images_with_kept_tiles, digits=0),
            format_value(metrics.get("tile_reduction")),
            format_value(tiles_per_image),
            format_value(kept_per_image),
            format_value(metrics.get("total_time")),
            format_value(wall_time),
            format_value(metrics.get("tile_fps")),
            format_value(metrics.get("real_fps")),
            format_value(metrics.get("precision")),
            format_value(metrics.get("recall")),
            format_value(metrics.get("map50")),
            format_value(metrics.get("map50_95")),
        ])
        script_results.append((label, metrics))

        if args.debug:
            print(f"\n===== RAW OUTPUT: {label} =====")
            print(output.rstrip())
            if stderr.strip():
                print(f"\n----- STDERR: {label} -----")
                print(stderr.rstrip())
            print("\n----- DEBUG SUMMARY -----")
            print(f"Images processed: {images_processed if images_processed is not None else '-'}")
            print(f"Tiles generated/image: {format_value(tiles_per_image)}")
            print(f"Tiles kept/image: {format_value(kept_per_image)}")
            if images_processed and tiles_kept is not None:
                keep_rate = tiles_kept / tiles_generated if tiles_generated else 0.0
                print(f"Keep rate: {keep_rate:.4f}")
            print(f"Reported eval time: {format_value(metrics.get('total_time'))} sec")
            print(f"Subprocess wall time: {wall_time:.3f} sec")

    print_table(rows)

    if args.debug and len(rows) == 2:
        print("\n===== DELTA CHECK =====")
        eval_1_metrics = script_results[0][1]
        eval_2_metrics = script_results[1][1]
        eval_1_images = eval_1_metrics.get("images_processed") or 0
        eval_2_images = eval_2_metrics.get("images_processed") or 0
        eval_1_kept = eval_1_metrics.get("tiles_kept") or 0
        eval_2_kept = eval_2_metrics.get("tiles_kept") or 0
        print(f"Evaluation 1 kept tiles/image: {eval_1_kept / eval_1_images:.4f}" if eval_1_images else "Evaluation 1 kept tiles/image: -")
        print(f"Evaluation 2 kept tiles/image: {eval_2_kept / eval_2_images:.4f}" if eval_2_images else "Evaluation 2 kept tiles/image: -")
        print(f"Kept-tile difference/image: {(eval_1_kept / eval_1_images) - (eval_2_kept / eval_2_images):.4f}" if eval_1_images and eval_2_images else "Kept-tile difference/image: -")


if __name__ == "__main__":
    main()
