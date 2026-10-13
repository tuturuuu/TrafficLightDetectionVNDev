"""Explicit paired-frame LTS training using the installed YOLO detection loss.

One original frame supervises relevance; one uniformly sampled candidate crop of
that SAME frame supervises detection. No gradient through hard tile selection.
"""

import copy
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .detectors import register_custom_modules
from .data import check_class_map, list_research_images, load_gt_strict
from .models.coarse_features import YOLOCoarseFeatureProvider, letterbox
from .models.lts import LTSHead
from .models.lts_checkpoint import MODES, save_lts_checkpoint
from .provenance import file_sha256, software_versions, source_fingerprints
from .relevance_targets import relevance_loss, relevance_targets
from .selectors.relevance_utils import score_tiles
from .tiling import make_grid


class PairedFrames(Dataset):
    def __init__(self, images, labels, nc, tile_size, overlap, seed=0):
        self.paths = [Path(p) for p in list_research_images(images)]
        if not self.paths:
            raise ValueError(f'No images in {images}')
        self.labels, self.nc = Path(labels), nc
        self.tile_size, self.overlap, self.seed, self.epoch = tile_size, overlap, seed, 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        label = self.labels / (path.stem + '.txt')
        if not label.is_file():
            raise FileNotFoundError(label)
        h, w = image.shape[:2]
        boxes = load_gt_strict(label, h, w, self.nc)
        tiles, _, _ = make_grid(h, w, self.tile_size, self.overlap)
        rng = random.Random(self.seed + self.epoch * len(self.paths) + index)
        tile = tiles[rng.randrange(len(tiles))]
        return {'image': image, 'boxes': boxes, 'tile': tile, 'path': str(path)}


def collate_frames(items):
    return items


def fine_training_batch(samples, imgsz, device, min_visibility=.3):
    arrays, boxes, classes, indices = [], [], [], []
    for index, sample in enumerate(samples):
        x1, y1, x2, y2 = sample['tile'][:4]
        rgb, geometry = letterbox(sample['image'][y1:y2, x1:x2], imgsz)
        arrays.append(rgb)
        for bx1, by1, bx2, by2, cls in sample['boxes']:
            ix1, iy1, ix2, iy2 = max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2)
            area = max(bx2-bx1, 0) * max(by2-by1, 0)
            if ix2 <= ix1 or iy2 <= iy1 or area <= 0 or (ix2-ix1)*(iy2-iy1) < min_visibility*area:
                continue
            a, b, c, d = geometry.transform_boxes([(ix1-x1, iy1-y1, ix2-x1, iy2-y1)])[0]
            boxes.append(((a+c)/(2*imgsz), (b+d)/(2*imgsz), (c-a)/imgsz, (d-b)/imgsz))
            classes.append(cls); indices.append(index)
    return {'img': torch.from_numpy(np.stack(arrays)).to(device).float() / 255,
            'bboxes': torch.tensor(boxes, device=device, dtype=torch.float32).reshape(-1, 4),
            'cls': torch.tensor(classes, device=device, dtype=torch.float32).reshape(-1, 1),
            'batch_idx': torch.tensor(indices, device=device, dtype=torch.long)}


class LTSSystem(nn.Module):
    def __init__(self, detector, feature_stage, coarse_imgsz, mode='joint', lambda_rel=1.,
                 pos_weight=1., fine_imgsz=640, min_visibility=.3):
        super().__init__()
        if mode not in MODES or not math.isfinite(lambda_rel) or lambda_rel <= 0:
            raise ValueError('Valid training mode and positive lambda_rel required')
        self.detector = detector
        self.mode, self.lambda_rel, self.pos_weight = mode, lambda_rel, pos_weight
        self.fine_imgsz, self.min_visibility = fine_imgsz, min_visibility
        for name, parameter in detector.named_parameters():
            parameter.requires_grad_(mode == 'joint' and '.dfl.' not in name)
        self.provider = YOLOCoarseFeatureProvider(detector, feature_stage, coarse_imgsz)
        self.head = LTSHead(self.provider.channels).to(next(detector.parameters()).device)

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'frozen_head_pilot':
            self.detector.eval()
        return self

    def losses(self, samples):
        images = [s['image'] for s in samples]
        boxes = [s['boxes'] for s in samples]
        coarse = self.provider(images, grad_enabled=self.mode == 'joint' and torch.is_grad_enabled(), profile=False)
        logits = self.head(coarse.features)
        targets, valid = relevance_targets(boxes, coarse.geometries, logits.shape[-2:], logits.device)
        rel = relevance_loss(logits, targets, valid, self.pos_weight)
        det = rel.new_zeros(())
        if self.mode == 'joint':
            batch = fine_training_batch(samples, self.fine_imgsz, logits.device, self.min_visibility)
            # Ultralytics returns three loss components multiplied by batch size.
            components, _ = self.detector.loss(batch)
            det = components.sum() / len(samples)
        return {'loss': det + self.lambda_rel * rel, 'det_loss': det, 'rel_loss': rel,
                'logits': logits, 'targets': targets, 'valid': valid, 'geometry': coarse.geometries}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def export_pair(system, directory, training_metadata):
    """Export one immutable matched detector/head pair, including frozen pilots."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    detector = copy.deepcopy(system.detector).cpu().float().eval()
    detector.criterion = None
    args = vars(detector.args) if hasattr(detector.args, '__dict__') else dict(detector.args)
    path = directory / 'detector.pt'
    torch.save({'model': detector, 'ema': None, 'train_args': args,
                'version': software_versions()['ultralytics']}, path)
    metadata = save_lts_checkpoint(directory / 'lts.pt', system.head, system.provider, path,
                                   system.mode, training_metadata)
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return path


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def train_lts(config, dry_run=False, resume=None, stop_after_epochs=None):
    """Config-driven trainer; existing run directories are accepted only for resume."""
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    cfg = copy.deepcopy(config)
    output = Path(cfg['out'])
    if output.exists() and resume is None:
        raise FileExistsError(f'{output} exists; use a new out or explicit --resume')
    if cfg['training_mode'] not in MODES:
        raise ValueError(f'training_mode must be one of {MODES}')
    tr = cfg['train']
    for key in ('epochs', 'batch_size', 'patience'):
        if not isinstance(tr[key], int) or isinstance(tr[key], bool) or tr[key] <= 0:
            raise ValueError(f'train.{key} must be a positive integer')
    for key in ('lr', 'lambda_rel', 'pos_weight', 'clip_grad'):
        if not math.isfinite(tr[key]) or tr[key] <= 0:
            raise ValueError(f'train.{key} must be finite and positive')
    if not 0 <= tr['threshold'] <= 1 or tr.get('warmup_steps', 100) < 0:
        raise ValueError('Invalid threshold or warmup_steps')
    tiling = cfg['tiling']
    if tiling['tile_size'] <= 0 or not 0 <= tiling['overlap'] < 1 or not 0 < tiling.get('min_visibility', .3) <= 1:
        raise ValueError('Invalid tile size, overlap or minimum visibility')
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError('--stop-after-epochs must be positive')
    if dry_run:
        return cfg
    seed_everything(tr['seed'])
    register_custom_modules()
    source = Path(cfg['detector']['weights'])
    if not source.is_file():
        raise FileNotFoundError(source)
    device = torch.device(tr['device'])
    detector = YOLO(str(source)).model.to(device).float()
    detector.args = get_cfg(overrides={**cfg.get('loss', {}), 'epochs': tr['epochs']})
    detector.criterion = None
    system = LTSSystem(detector, **cfg['features'], mode=cfg['training_mode'],
        lambda_rel=tr['lambda_rel'], pos_weight=tr['pos_weight'], fine_imgsz=cfg['detector']['imgsz'],
        min_visibility=cfg['tiling'].get('min_visibility', .3))
    data = cfg['dataset']
    check_class_map(data.get('data_yaml'), detector.names)
    manifest = Path(data['split_manifest'])
    metadata = {'config': cfg, 'split_manifest_sha256': file_sha256(manifest),
                'source_detector_sha256': file_sha256(source), 'seed': tr['seed'],
                'source_sha256': source_fingerprints(), 'software': software_versions(),
                'class_names': detector.names,
                'data_yaml_sha256': file_sha256(data['data_yaml']) if data.get('data_yaml') else None,
                'sampling': 'one_uniform_candidate_crop_per_original_frame_per_epoch',
                'loss_reduction': 'sum_YOLO_components_per_image + lambda * valid_cell_mean_BCE',
                'ema': False, 'status': 'experimental'}
    datasets = {split: PairedFrames(data[f'{split}_images'], data[f'{split}_labels'],
        detector.model[-1].nc, cfg['tiling']['tile_size'], cfg['tiling']['overlap'], tr['seed'])
        for split in ('train', 'val')}
    generator = torch.Generator().manual_seed(tr['seed'])
    loaders = {split: DataLoader(ds, batch_size=tr['batch_size'], shuffle=split == 'train',
        num_workers=tr.get('workers', 0), collate_fn=collate_frames, generator=generator)
        for split, ds in datasets.items()}
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],
        lr=tr['lr'], weight_decay=tr.get('weight_decay', 0.0005))
    steps = max(1, len(loaders['train']) * tr['epochs'])
    warmup = min(tr.get('warmup_steps', 100), steps)

    def lr_factor(step):
        if warmup and step < warmup:
            return max(1, step + 1) / warmup
        return .01 + .99 * (1 + math.cos(math.pi * min(1, (step-warmup) / max(steps-warmup, 1)))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = bool(tr.get('amp', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best, bad_epochs, history = 0, float('inf'), 0, []
    if resume is not None:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['metadata'] != metadata:
            raise ValueError('Resume config, detector or split fingerprint differs')
        system.load_state_dict(state['system'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']); generator.set_state(state['generator'])
        start, best, bad_epochs, history = state['epoch'] + 1, state['best'], state['bad_epochs'], state['history']
        if system.mode == 'joint':
            detector.criterion = detector.init_criterion()
            for _ in range(state['criterion_updates']):
                if hasattr(detector.criterion, 'update'):
                    detector.criterion.update()
        _restore_rng(state['rng'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    for epoch in range(start, tr['epochs']):
        row = {'epoch': epoch + 1}
        for split, loader in loaders.items():
            training = split == 'train'
            datasets[split].epoch = epoch if training else 0
            system.train(training)
            totals = {'loss': 0., 'det_loss': 0., 'rel_loss': 0.}
            seen, tp, positives, kept, candidates, positive_cells, valid_cells = 0, 0, 0, 0, 0, 0, 0
            for samples in loader:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
                    values = system.losses(samples)
                if not torch.isfinite(values['loss']):
                    raise FloatingPointError('Nonfinite LTS training loss')
                if training:
                    scaler.scale(values['loss']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(system.parameters(), tr.get('clip_grad', 10.))
                    scaler.step(optimizer); scaler.update(); scheduler.step()
                for key in totals:
                    totals[key] += float(values[key].detach()) * len(samples)
                seen += len(samples)
                probs = torch.sigmoid(values['logits'].detach()).float().cpu().numpy()
                target = values['targets'].bool(); mask = values['valid'].bool()
                pred = values['logits'].detach().sigmoid() > tr['threshold']
                tp += int((pred & target & mask).sum()); positives += int((target & mask).sum())
                positive_cells += int((target & mask).sum()); valid_cells += int(mask.sum())
                for sample, heat, geometry in zip(samples, probs[:, 0], values['geometry']):
                    tiles, _, _ = make_grid(*sample['image'].shape[:2], cfg['tiling']['tile_size'], cfg['tiling']['overlap'])
                    scores, _ = score_tiles(heat, tiles, geometry)
                    kept += int((scores > tr['threshold']).sum()); candidates += len(tiles)
            row[split] = {**{k: v / max(seen, 1) for k, v in totals.items()},
                          'cell_recall': tp / max(positives, 1), 'positive_fraction': positive_cells / max(valid_cells, 1),
                          'kept_tile_fraction': kept / max(candidates, 1)}
        criterion = getattr(detector, 'criterion', None)
        if hasattr(criterion, 'update'):
            criterion.update()
        improved = row['val']['loss'] < best
        if improved:
            best, bad_epochs = row['val']['loss'], 0
            pair = output / f'best_epoch{epoch + 1:04d}'
            export_pair(system, pair, {**metadata, 'epoch': epoch + 1})
            (output / 'best.json').write_text(json.dumps({'directory': pair.name, 'epoch': epoch + 1, 'val_loss': best}, indent=2))
        else:
            bad_epochs += 1
        history.append(row)
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        state = {'metadata': metadata, 'system': system.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'generator': generator.get_state(),
                 'rng': _rng_state(), 'epoch': epoch, 'best': best, 'bad_epochs': bad_epochs, 'history': history,
                 'criterion_updates': getattr(criterion, 'updates', 0)}
        temporary = output / 'resume.tmp.pt'
        torch.save(state, temporary); temporary.replace(output / 'resume.pt')
        print(json.dumps(row), flush=True)
        if bad_epochs >= tr.get('patience', 10) or (stop_after_epochs and epoch + 1 - start >= stop_after_epochs):
            break
    return history
