import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

ROOT = Path(__file__).resolve().parent

sys.path.append(str(ROOT / "utils"))
from custom_modules import CBAM, SE

tasks.CBAM = CBAM
tasks.SE = SE

SEEDS = [100]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(ROOT / "data/SeaDroneSees_70_15_15/data_tiled.yaml"),
    )
    parser.add_argument(
        "--name-prefix",
        default="seadronesees_yolov8_cbam_tiled_default",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def train_one(seed, args):
    print(f"\n=== Training seed {seed} ===")
    print("CUDA:", torch.cuda.is_available())
    print("Data:", args.data)

    model = YOLO(str(ROOT / "yolo_config/YOLOv8_CBAM.yaml"))

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "runs"),
        name=f"{args.name_prefix}_seed{seed}",
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=seed,
        deterministic=True,
    )

    metrics = model.val()
    print(f"Seed {seed} metrics:", metrics)
    return results


def main():
    args = parse_args()
    for seed in args.seeds:
        train_one(seed, args)

if __name__ == "__main__":
    main()