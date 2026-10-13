#!/usr/bin/env python
"""Train a frozen LTS pilot or joint detector+LTS; no implicit experiment sweep."""

import argparse
import json

from omegaconf import OmegaConf

from adaptile.training_lts import train_lts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('overrides', nargs='*')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', help='resume.pt from the SAME run and resolved config')
    parser.add_argument('--stop-after-epochs', type=int, help='stop early for an engineering smoke run; preserves the planned schedule')
    args = parser.parse_intermixed_args()
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(args.config),
                                    OmegaConf.from_dotlist(args.overrides)), resolve=True)
    print(json.dumps(config, indent=2))
    train_lts(config, dry_run=args.dry_run, resume=args.resume, stop_after_epochs=args.stop_after_epochs)


if __name__ == '__main__':
    main()
