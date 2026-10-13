"""LAT-YOLO v1 experiment protocol, opt-in alongside historical evaluation."""

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from .data import check_class_map, label_path_for, list_research_images, load_gt_strict
from .experiment import detailed_records, run_provenance, write_visuals
from .metrics import evaluate
from .metrics_coco import evaluate_coco
from .pipeline import Pipeline, run_dataset
from .provenance import architecture_sha256, file_sha256
from .registry import DETECTORS, MERGERS, SELECTORS, TILERS


def run_evaluation(cfg):
    if cfg.warmup_mode != 'full':
        raise ValueError('lat_yolo_v1 requires full pipeline warmup')
    destination = Path(cfg.out)
    sidecars = [destination.with_suffix('.yaml'), destination.with_suffix('.predictions.json')]
    if destination.exists() or any(p.exists() for p in sidecars):
        raise FileExistsError('Research outputs are immutable; choose a new eval.out')
    paths = list_research_images(cfg.dataset.images, cfg.max_images)
    if not paths:
        raise ValueError(f'No images in {cfg.dataset.images}')
    images, gts = [], []
    for path in paths:
        image = cv2.imread(path)
        label = Path(label_path_for(path, cfg.dataset.labels))
        if image is None or not label.is_file():
            raise ValueError(f'Unreadable image or missing annotation: {path}, {label}')
        images.append(image)
        gts.append(load_gt_strict(label, *image.shape[:2]))
    provenance = run_provenance(cfg, paths)
    if provenance['detector_sha256'] is None:
        raise FileNotFoundError('LAT evaluation requires an existing local detector checkpoint')
    detector = DETECTORS.build(cfg.detector)
    check_class_map(cfg.dataset.get('data_yaml'), detector.model.names)
    nc = len(detector.model.names)
    if any(box[4] >= nc for boxes in gts for box in boxes):
        raise ValueError('Ground-truth class is outside the detector class map')
    merger = MERGERS.build(cfg.merger)
    detector_params = sum(p.numel() for p in detector.model.model.parameters())
    arch_hash = architecture_sha256(detector.model.model)
    rows, records_by_method = [], []
    sizes = cfg.tile_size if cfg.tiler.name != 'none' else [None]
    for size in sizes:
        tiler_config = dict(cfg.tiler)
        if size is not None:
            tiler_config['tile_size'] = size
        tiler = TILERS.build(tiler_config)
        for name, selector_cfg in cfg.selectors.items():
            selector = SELECTORS.build(selector_cfg)
            pipe = Pipeline(tiler, selector, detector, merger, profile_stages=cfg.profile_stages,
                            scoped_sync=True)
            run = run_dataset(pipe, images, cfg.n_warmup, warmup_mode='full', retain_details=True)
            operating = evaluate(run.preds, gts)
            accuracy = evaluate_coco(run.preds, gts, [im.shape[:2] for im in images], cfg.coco_max_dets) \
                if cfg.metric_backend == 'coco' else operating
            precision, recall = operating['precision'], operating['recall']
            n = len(images)
            mode = ('uniform' if size else 'notile') if name == 'none' else name
            records = detailed_records(run, paths, gts, cfg)
            covered, total = sum(x['covered_gt'] for x in records), sum(x['n_gt'] for x in records)
            selector_config = OmegaConf.to_container(selector_cfg, resolve=True)
            weights = selector_config.get('weights')
            meta = {**provenance, 'architecture_sha256': arch_hash,
                    'class_names': detector.model.names,
                    'selector': selector_config, 'selector_sha256': file_sha256(weights) if weights else None,
                    'feature': getattr(getattr(selector, 'provider', None), 'metadata', None),
                    'lts_checkpoint': getattr(selector, 'checkpoint_metadata', None),
                    'intra_tile_predict_options': detector.predict_options,
                    'precision_recall_protocol': 'micro_IoU0.5_at_detector_conf',
                    'memory_scope': 'current_process_detector_and_one_selector'}
            stages = sorted({key for r in records for key in r['timings_ms']})
            stage_summary = {key: {'mean': float(np.mean([r['timings_ms'].get(key, 0.) for r in records])),
                                  'p50': float(np.percentile([r['timings_ms'].get(key, 0.) for r in records], 50)),
                                  'p95': float(np.percentile([r['timings_ms'].get(key, 0.) for r in records], 95))}
                             for key in stages}
            row = {'schema_version': 'lat_yolo_v1', 'tile_size': size, 'method': mode,
                   'n_images': n, 'tiles_per_img': run.total_tiles/n, 'kept_per_img': run.kept_tiles/n,
                   'fine_forward_batches': sum((r.n_kept + detector.batch - 1)//detector.batch
                                               for r in run.image_results),
                   'coarse_backbone_calls': n if getattr(selector, 'provider', None) is not None else 0,
                   'tile_reduction': 1-run.kept_tiles/max(run.total_tiles, 1),
                   'mAP50': accuracy['mAP50'], 'mAP50-95': accuracy['mAP50-95'],
                   'AP_S': accuracy.get('AP_S'), 'precision': precision, 'recall': recall,
                   'f1': 2*precision*recall/max(precision+recall, 1e-12),
                   'fps': n/max(run.wall_s, 1e-12), 'latency_mean_ms': 1000*run.wall_s/n,
                   'time_min': run.wall_s/60, 'wall_s': run.wall_s, 'select_s': run.select_s,
                   'detect_s': run.detect_s, 'merge_s': run.merge_s, 'stage_latency_ms': stage_summary,
                   'selector_gt_coverage': covered/total if total else None,
                   'coverage_visibility': cfg.coverage_visibility,
                   'peak_gpu_memory_mb': run.peak_gpu_memory_mb,
                   'detector_parameters_loaded': detector_params,
                   'relevance_head_parameters': sum(p.numel() for p in selector.head.parameters())
                       if getattr(selector, 'head', None) is not None else 0 if name == 'param_free_avg' else None,
                   'provenance': meta}
            if cfg.metric_backend == 'coco':
                row['coco_protocol'] = {k: accuracy[k] for k in ('protocol', 'max_dets', 'area_source')}
            if 'agnostic' in cfg.metrics:
                row['agnostic'] = evaluate(run.preds, gts, class_aware=False)
            rows.append(row)
            if cfg.export_predictions:
                records_by_method.append({'tile_size': size, 'method': mode, 'images': records})
            if cfg.visualize:
                write_visuals(destination.parent / f'{destination.stem}_{size}_{mode}_visuals',
                              images, run.image_results, cfg.visualize, gts)
            print(f'{mode} tile={size} kept={row["kept_per_img"]:.2f} FPS={row["fps"]:.2f} '
                  f'mAP50={row["mAP50"]} recall={recall:.4f}', flush=True)
            del pipe, selector, run
            gc.collect()
    serialized_rows = json.dumps(rows, indent=2, allow_nan=False)
    serialized_records = json.dumps(records_by_method, indent=2, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        handle.write(serialized_rows)
    OmegaConf.save(cfg.source, destination.with_suffix('.yaml'))
    if cfg.export_predictions:
        with destination.with_suffix('.predictions.json').open('x') as handle:
            handle.write(serialized_records)
    return rows
