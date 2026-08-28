import os
import sys

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks


sys.path.append(os.path.abspath("./utils"))
from custom_modules import CBAM, SE


tasks.CBAM = CBAM
tasks.SE = SE

SEEDS = [0, 42, 100]


def train_one(seed):
    print(f"\n=== Training no-tiling CBAM seed {seed} ===")
    print("CUDA:", torch.cuda.is_available())

    model = YOLO("yolo_config/YOLOv8_CBAM.yaml")

    results = model.train(
        data="/home/anhld/AdaptiveTiling/TrafficLightDetectionVNDev/data/SeaDroneSees_70_15_15/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs",
        name=f"seadronesees_yolov8_cbam_notiled_seed{seed}",
        device=0,
        workers=8,
        patience=30,
        seed=seed,
        deterministic=True,
    )

    metrics = model.val()
    print(f"Seed {seed} metrics:", metrics)
    return results


def main():
    for seed in SEEDS:
        train_one(seed)


if __name__ == "__main__":
    main()
