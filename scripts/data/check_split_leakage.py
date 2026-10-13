#!/usr/bin/env python3
"""Audit video overlap and create deterministic grouped 70/15/15 YOLO splits.

Kayuan: first 14 timestamp digits (YYYYMMDDhhmmss), removing underscore.
SeaDroneSee: source.drone/folder_name/video from original COCO metadata.
No frame is assigned independently; source data is never modified.
"""
import argparse
import itertools
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

SPLITS = ('train', 'val', 'test')
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def sequence_key(stem):
    compact = stem.replace('_', '')
    if not re.fullmatch(r'\d{15,}', compact):
        raise ValueError(f'Unrecognized Kayuan frame: {stem}')
    return compact[:14]


def collect(root, kind, metadata_root=None, allow_missing=False):
    metadata = {}
    if kind == 'seadrone':
        for path in (metadata_root or root).glob('annotations/instances_*.json'):
            for image in json.loads(path.read_text())['images']:
                source = image.get('source', {})
                if source.get('video') or source.get('image_name'):
                    metadata[Path(image['file_name']).name] = '/'.join(str(source.get(k, '')) for k in ('drone', 'folder_name', 'video')) # Still images grouped conservatively by capture folder
        if not metadata:
            raise ValueError('No video metadata; supply --metadata-root with original COCO annotations')
    samples = []
    for split in SPLITS:
        images = root / split / 'images' if (root / split / 'images').is_dir() else root / 'images' / split
        labels = root / split / 'labels' if (root / split / 'labels').is_dir() else root / 'labels' / split
        for image in sorted(images.glob('*')):
            if image.suffix.lower() not in EXTS:
                continue
            label = labels / (image.stem + '.txt')
            # Ignore unlabeled official SeaDroneSee test, never use it in resplitting.
            if kind == 'seadrone' and split == 'test' and not label.exists():
                continue
            if not label.is_file():
                if allow_missing:
                    continue
                raise FileNotFoundError(label)
            key = sequence_key(image.stem) if kind == 'kayuan' else metadata[image.name]
            samples.append((split, key, image, label))
    if not samples:
        raise ValueError(f'No labeled images: {root}')
    names = [s[2].name for s in samples]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate image names across splits')
    return samples


def audit(samples):
    groups = {s: {x[1] for x in samples if x[0] == s} for s in SPLITS}
    overlaps = {f'{a}/{b}': sorted(groups[a] & groups[b]) for a, b in itertools.combinations(SPLITS, 2)}
    return {'counts': {s: sum(x[0] == s for x in samples) for s in SPLITS},
            'sequence_counts': {s: len(groups[s]) for s in SPLITS},
            'overlap': overlaps, 'leakage': any(overlaps.values())}


def grouped_split(samples, seed=42):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample[1]].append(sample)
    if len(groups) < 3:
        raise ValueError('At least three videos required')
    items = list(sorted(groups.items()))
    random.Random(seed).shuffle(items)
    items.sort(key=lambda x: -len(x[1]))
    targets = dict(zip(SPLITS, (len(samples)*.7, len(samples)*.15, len(samples)*.15)))
    result = {s: [] for s in SPLITS}
    for key, members in items:
        split = max(SPLITS, key=lambda s: (targets[s]-len(result[s])) / targets[s])
        result[split].extend((split, key, img, label) for _, _, img, label in sorted(members, key=lambda x: str(x[2])))
    if any(not result[s] for s in SPLITS):
        raise ValueError('Unable to create three nonempty grouped splits')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--kind', choices=('kayuan', 'seadrone'), required=True)
    p.add_argument('--metadata-root', type=Path)
    p.add_argument('--output-root', type=Path, help='Create a new symlink dataset; refuses existing output')
    p.add_argument('--report', type=Path)
    p.add_argument('--exclude-missing-labels', action='store_true', help='Explicitly exclude images without labels, never treat them as negative')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    samples = collect(args.root, args.kind, args.metadata_root, args.exclude_missing_labels)
    report = {'source': str(args.root), 'excluded_missing_labels': [str(img) for split in SPLITS for img in ((args.root / split / 'images') if args.kind == 'kayuan' else (args.root / 'images' / split)).glob('*') if img.suffix.lower() in EXTS and not (((args.root / split / 'labels') if args.kind == 'kayuan' else (args.root / 'labels' / split)) / (img.stem + '.txt')).exists()], 'before': audit(samples)}
    if args.output_root:
        result = grouped_split(samples, args.seed)
        root = args.output_root.resolve()
        root.mkdir(parents=True, exist_ok=False)
        manifest = []
        for split, members in result.items():
            for folder in ('images', 'labels'):
                (root / split / folder).mkdir(parents=True)
            for _, key, image, label in members:
                for folder, src in (('images', image), ('labels', label)):
                    (root / split / folder / src.name).symlink_to(src.resolve())
                manifest.append({'split': split, 'sequence': key, 'image': str(image.resolve())})
        data = yaml.safe_load((args.root / 'data.yaml').read_text())
        data = {'path': str(root), **{s: f'{s}/images' for s in SPLITS}, 'names': data['names']}
        (root / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
        (root / 'split_manifest.json').write_text(json.dumps({'seed': args.seed, 'samples': manifest}, indent=2))
        report['after'] = audit([x for members in result.values() for x in members])
        (root / 'split_audit.json').write_text(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return int(report.get('after', report['before'])['leakage'])


if __name__ == '__main__':
    raise SystemExit(main())
