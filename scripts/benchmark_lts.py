#!/usr/bin/env python
"""Same original-image LAT evaluation, with full warmup and explicit profiling mode."""

import argparse

from omegaconf import OmegaConf

from adaptile.config import load_config
from adaptile.evaluation_lts import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('overrides', nargs='*')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_intermixed_args()
    cfg = load_config(args.config, args.overrides)
    if cfg.protocol != 'lat_yolo_v1':
        raise ValueError('Use eval.protocol=lat_yolo_v1 and eval.warmup_mode=full')
    print(OmegaConf.to_yaml(cfg.source))
    if not args.dry_run:
        run_evaluation(cfg)


if __name__ == '__main__':
    main()
