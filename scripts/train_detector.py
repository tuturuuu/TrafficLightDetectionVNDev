#!/usr/bin/env python
"""
Train detector (YOLO + attention, có/không tiling) từ MỘT file config.
Thay cho legacy/train_seadronesee_*.py, legacy/kayuan2024/, legacy/bosch/.

    python scripts/train_detector.py configs/train/detector_kayuan.yaml
    python scripts/train_detector.py configs/train/detector_kayuan.yaml model.attention=none tiling.type=none
    python scripts/train_detector.py configs/train/detector_kayuan.yaml train.seeds=[0] train.epochs=1 --dry-run

tiling.type=uniform: cắt train/val thành tile (nếu chưa có) rồi train trên tile.
Mỗi seed một run: <output.project>/<output.name>_seed<seed>/weights/best.pt.
"""

import argparse

from omegaconf import OmegaConf

from adaptile.training import train_detector


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="vd configs/train/detector_kayuan.yaml")
    p.add_argument("overrides", nargs="*", help="key=value, vd train.epochs=50 model.attention=se")
    p.add_argument("--dry-run", action="store_true", help="chỉ in tham số, không cắt tile, không train")
    p.add_argument("--overwrite", action="store_true", help="cho phép ghi vào run đã có")
    args = p.parse_intermixed_args()   # cho phép đặt --cờ ở bất kỳ đâu

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    print(OmegaConf.to_yaml(cfg))
    runs = train_detector(cfg, dry_run=args.dry_run, overwrite=args.overwrite)
    if not args.dry_run:
        for r in runs:
            OmegaConf.save(cfg, f"{r}/train_config.yaml")
        print("\nXong:", *[f"{r}/weights/best.pt" for r in runs], sep="\n  ")


if __name__ == "__main__":
    main()
