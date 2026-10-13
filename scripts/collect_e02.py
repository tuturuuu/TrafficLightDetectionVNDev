#!/usr/bin/env python
"""Mean +/- sample SD for matched LAT-YOLO result JSONs (default seeds 0/42/100)."""

import argparse
import csv
import json
from pathlib import Path

from adaptile.report_lts import METRICS, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', help='explicit JSON paths or shell-expanded glob')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 42, 100])
    parser.add_argument('--out', required=True, help='new .json path; a .csv sidecar is also written')
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('--seeds must be distinct')
    destination = Path(args.out)
    if destination.suffix != '.json':
        parser.error('--out must end in .json')
    csv_path = destination.with_suffix('.csv')
    if destination.exists() or csv_path.exists():
        raise FileExistsError('Choose a new report output path')
    summary = aggregate(args.results, tuple(args.seeds))
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        handle.write(serialized + '\n')
    fields = ['experiment_id', 'dataset', 'tile_size', 'method', 'training_mode', *METRICS]
    with csv_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({**{k: row[k] for k in fields[:5]},
                             **{k: row['metrics'][k]['display'] for k in METRICS}})
    print(f'Saved {destination} and {csv_path}; SD uses ddof=1.')


if __name__ == '__main__':
    main()
