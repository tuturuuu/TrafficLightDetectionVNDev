"""Versioned evaluation records; accuracy and speed always refer to original images."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from .provenance import file_sha256, software_versions, source_fingerprints


def coverage(boxes, tiles, min_visibility=.5):
    if not 0 < min_visibility <= 1:
        raise ValueError('coverage_visibility must be in (0, 1]')
    covered = 0
    for x1, y1, x2, y2, *_ in boxes:
        area = (x2-x1)*(y2-y1)
        if area <= 0:
            raise ValueError('GT has nonpositive area')
        covered += any(max(0, min(x2, t[2])-max(x1, t[0])) *
                       max(0, min(y2, t[3])-max(y1, t[1])) / area >= min_visibility for t in tiles)
    return covered, len(boxes)


def dataset_fingerprint(paths, labels):
    """Ordered image/label content hashes, not a leakage or split audit."""
    h = hashlib.sha256()
    for path in paths:
        p = Path(path)
        label = Path(labels) / (p.stem + '.txt')
        if not p.is_file() or not label.is_file():
            raise FileNotFoundError(p if not p.is_file() else label)
        h.update(json.dumps([p.name, file_sha256(p), file_sha256(label)]).encode())
    return h.hexdigest()


def run_provenance(cfg, paths):
    def git(*args):
        p = subprocess.run(['git', *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    manifest = cfg.dataset.get('split_manifest')
    weights = cfg.detector.get('weights')
    return {'dataset': cfg.dataset.name, 'seed': cfg.seed,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'experiment_id': cfg.source.eval.get('experiment_id'),
            'source_sha256': source_fingerprints(),
            'code_commit': git('rev-parse', 'HEAD'), 'code_status': git('status', '--short'),
            'code_diff_sha256': hashlib.sha256((git('diff') or '').encode()).hexdigest(),
            'software': software_versions(), 'cuda_build': torch.version.cuda,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'device': cfg.detector.get('device'),
            'detector_sha256': file_sha256(weights) if weights and Path(weights).is_file() else None,
            'data_yaml_sha256': file_sha256(cfg.dataset.data_yaml) if cfg.dataset.get('data_yaml') else None,
            'split_manifest_sha256': file_sha256(manifest) if manifest else None,
            'ordered_images_labels_sha256': dataset_fingerprint(paths, cfg.dataset.labels),
            'n_images': len(paths), 'disk_io_in_timing': False,
            'warmup_mode': cfg.warmup_mode, 'n_warmup': cfg.n_warmup,
            'profile_stages': cfg.profile_stages, 'metric_backend': cfg.metric_backend,
            'protocol': cfg.protocol, 'detector_conf': cfg.detector.conf,
            'fine_imgsz': cfg.detector.imgsz, 'fine_batch': cfg.detector.batch,
            'overlap': cfg.tiler.get('overlap'), 'merge': dict(cfg.merger)}


def detailed_records(run, paths, gts, cfg):
    records = []
    for path, gt, result in zip(paths, gts, run.image_results):
        covered, total = coverage(gt, result.kept_tiles, cfg.coverage_visibility)
        item = {'image': str(path), 'predictions': [list(map(float, d[:5])) + [int(d[5])] for d in result.dets],
                'n_candidate_tiles': result.n_tiles, 'n_selected_tiles': result.n_kept,
                'selected_tiles': [list(t) for t in result.kept_tiles],
                'timings_ms': result.timings_ms, 'covered_gt': covered, 'n_gt': total}
        if result.selection is not None:
            s = result.selection
            item.update(tile_scores=s.scores.tolist(), feature_shape=list(s.relevance_map.shape),
                        feature_geometry=asdict(s.geometry), geometry_fallback_count=s.fallback_count)
        records.append(item)
    return records


def write_visuals(directory, images, results, limit, ground_truths=None):
    """Validation inspection only; the heatmap is cropped out of letterbox padding."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for index, (image, result) in enumerate(zip(images[:limit], results[:limit])):
        overlay = image.copy()
        if result.selection is not None:
            selection = result.selection
            h, w = selection.geometry.input_hw
            relevance = cv2.resize(selection.relevance_map, (w, h), interpolation=cv2.INTER_NEAREST)
            left, top = selection.geometry.pad_xy
            rh, rw = selection.geometry.resized_hw
            relevance = relevance[top:top+rh, left:left+rw]
            heat = cv2.applyColorMap((np.clip(relevance, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.resize(heat, (image.shape[1], image.shape[0]))
            overlay = cv2.addWeighted(overlay, .65, heat, .35, 0)
        for x1, y1, x2, y2, *_ in result.kept_tiles:
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        if ground_truths is not None:
            for x1, y1, x2, y2, cls in ground_truths[index]:
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
                cv2.putText(overlay, str(cls), (int(x1), max(10, int(y1)-2)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        if not cv2.imwrite(str(directory / f'{index:05d}.jpg'), overlay):
            raise IOError('Failed to write relevance visualization')
