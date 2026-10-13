"""Aggregate versioned LAT results without mixing seeds or incompatible protocols."""

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = ('tiles_per_img', 'kept_per_img', 'tile_reduction', 'mAP50', 'mAP50-95',
           'AP_S', 'precision', 'recall', 'f1', 'fps', 'latency_mean_ms', 'time_min',
           'wall_s', 'select_s', 'detect_s', 'merge_s', 'selector_gt_coverage')
PROTOCOL_KEYS = ('dataset', 'ordered_images_labels_sha256', 'split_manifest_sha256', 'data_yaml_sha256',
                 'architecture_sha256', 'class_names', 'metric_backend', 'protocol',
                 'n_images', 'warmup_mode', 'n_warmup', 'profile_stages',
                 'disk_io_in_timing', 'fine_imgsz', 'fine_batch', 'detector_conf',
                 'overlap', 'merge', 'intra_tile_predict_options', 'software',
                 'source_sha256', 'device', 'gpus', 'cuda_build')


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def aggregate(paths, expected_seeds=(0, 42, 100)):
    groups, controls = defaultdict(dict), {}
    for path in paths:
        rows = json.loads(Path(path).read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'Expected nonempty LAT result list: {path}')
        for row in rows:
            if row.get('schema_version') != 'lat_yolo_v1':
                raise ValueError(f'Legacy/unrecognized schema cannot enter LAT report: {path}')
            meta = row['provenance']
            experiment = meta.get('experiment_id')
            seed = meta.get('seed')
            if not experiment or seed not in expected_seeds:
                raise ValueError('Set a shared eval.experiment_id and an expected eval.seed for every run')
            missing = [k for k in PROTOCOL_KEYS if k not in meta]
            if missing or not meta['split_manifest_sha256'] or not meta['detector_sha256']:
                raise ValueError(f'Incomplete experiment provenance: {missing}')
            key = (experiment, meta['dataset'], row['tile_size'], row['method'])
            if seed in groups[key]:
                raise ValueError(f'Duplicate seed {seed} for {key}; do not mix threshold sweeps/repeats')
            common = {k: meta[k] for k in PROTOCOL_KEYS}
            common['coco_protocol'] = row.get('coco_protocol')
            common['coverage_visibility'] = row['coverage_visibility']
            # Within a paired selector comparison, weights must be identical for each seed.
            control_key = (experiment, meta['dataset'], seed, row['tile_size'])
            control = _canonical([common, meta['detector_sha256']])
            if control_key in controls and controls[control_key] != control:
                raise ValueError(f'Unmatched detector/protocol across selectors: {control_key}')
            controls[control_key] = control
            selector = {k: v for k, v in meta['selector'].items() if k != 'weights'}
            checkpoint = meta.get('lts_checkpoint') or {}
            training = checkpoint.get('training', {}).get('config', {})
            recipe = {k: v for k, v in training.get('train', {}).items() if k != 'seed'}
            signature = _canonical([common, selector, checkpoint.get('training_mode'), recipe])
            groups[key][seed] = (row, str(path), signature)
    summary = []
    for key, by_seed in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if set(by_seed) != set(expected_seeds):
            raise ValueError(f'{key}: expected seeds {expected_seeds}, found {sorted(by_seed)}')
        signatures = {value[2] for value in by_seed.values()}
        if len(signatures) != 1:
            raise ValueError(f'{key}: incompatible settings across seeds')
        entries = [by_seed[s][0] for s in expected_seeds]
        metrics = {}
        for name in METRICS:
            values = [row[name] for row in entries]
            if all(v is None for v in values):
                metrics[name] = {'mean': None, 'std': None, 'n': 0, 'display': 'not defined'}
                continue
            if any(v is None or not math.isfinite(v) for v in values):
                raise ValueError(f'{key}: incomplete/nonfinite metric {name}')
            avg, sd = mean(values), stdev(values) if len(values) > 1 else 0.
            metrics[name] = {'mean': avg, 'std': sd, 'n': len(values), 'display': f'{avg:.4f} +/- {sd:.4f}'}
        summary.append({'experiment_id': key[0], 'dataset': key[1], 'tile_size': key[2],
                        'method': key[3], 'seeds': list(expected_seeds),
                        'training_mode': (entries[0]['provenance'].get('lts_checkpoint') or {}).get('training_mode'),
                        'std_definition': 'sample standard deviation (ddof=1), not confidence interval',
                        'metrics': metrics, 'sources': [by_seed[s][1] for s in expected_seeds]})
    return summary
