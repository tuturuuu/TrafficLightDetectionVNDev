#!/usr/bin/env python
"""
Train tile selector (GridNet hoặc CNN từng tile) từ MỘT file config.
Thay cho `grid_proposal_net.py train` (cũ) và `legacy/cnn_classifier/small_cnn.py`.

    python scripts/train_selector.py configs/train/selector_kayuan.yaml
    python scripts/train_selector.py configs/train/selector_kayuan.yaml gridnet.pos_weight=8 out=weights/gridnet_pw8.pth
    python scripts/train_selector.py configs/train/selector_kayuan.yaml type=tile_cnn out=weights/tile_cnn_v2.pth
"""

import argparse
import os
import sys

from omegaconf import OmegaConf

from adaptile.training import train_selector


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="vd configs/train/selector_kayuan.yaml")
    p.add_argument("overrides", nargs="*", help="key=value, vd type=tile_cnn gridnet.epochs=60")
    p.add_argument("--dry-run", action="store_true", help="chỉ in tham số, không train")
    p.add_argument("--overwrite", action="store_true", help="cho phép ghi đè file out")
    args = p.parse_intermixed_args()   # cho phép đặt --cờ ở bất kỳ đâu

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    print(OmegaConf.to_yaml(cfg))
    if os.path.exists(cfg.out) and not args.overwrite:
        sys.exit(f"{cfg.out} đã tồn tại; đổi out=... hoặc thêm --overwrite")
    train_selector(cfg, dry_run=args.dry_run)
    if not args.dry_run:
        OmegaConf.save(cfg, os.path.splitext(cfg.out)[0] + ".yaml")
        print(f"Xong: {cfg.out}")


if __name__ == "__main__":
    main()
