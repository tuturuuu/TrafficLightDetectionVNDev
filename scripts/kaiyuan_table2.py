#!/usr/bin/env python
"""Prepare, train, and audit Table II assets; never execute test evaluation.

Run with the LAT-YOLO conda environment. GPU visibility is pinned by the launcher.
"""
import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('configs/experiments/kaiyuan_table2')
STATE = Path('outputs/lat_yolo/kaiyuan_cd/table2_train')
RUNS = Path('runs/lat_yolo/kaiyuan')
DATA = Path('data/kaiyuan_cd')
SEEDS = (0, 42, 100)
GPU_UUID = 'GPU-7459ff17-edbb-d846-2f31-c5af0ae49739'
ROWS = (
    ('A1_baseline_se', 'none', 'none', 'se'),
    ('A2_fixed', 'uniform', 'none', 'none'),
    ('A3_naive', 'uniform', 'tile_cnn', 'none'),
    ('A4_avg', 'uniform', 'param_free_avg', 'none'),
    ('A5_lts', 'uniform', 'lts', 'none'),
    ('A6_fixed_cbam', 'uniform', 'none', 'cbam'),
    ('A7_fixed_se', 'uniform', 'none', 'se'),
    ('A8_lts_cbam', 'uniform', 'lts', 'cbam'),
    ('A9_lts_se', 'uniform', 'lts', 'se'),
)


def read(path):
    return yaml.safe_load(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def write_config(path, cfg):
    path = Path(path)
    if path.exists():
        if read(path) != cfg:
            raise ValueError(f'Refusing to change existing config: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def detector_run(attention, seed, tiled=True):
    return RUNS / f'yolo26m_p2_{"tiled" if tiled else "notile"}_{attention}_seed{seed}'


def lts_run(attention, seed):
    return RUNS / f'lts_joint_{attention}_pw1_lam1_seed{seed}'


def jobs():
    """Reuse detectors/CNN and the user's SE baseline; train only missing LTS."""
    items = []
    for attention in ('none', 'cbam', 'se'):
        for seed in SEEDS:
            items.append(dict(kind='detector', attention=attention, seed=seed,
                              out=str(detector_run(attention, seed)), tiled=True))
    for seed in SEEDS:
        items.append(dict(kind='selector', attention='none', seed=seed,
                          out=f'weights/lat_yolo/kaiyuan_cd/tilecnn_seed{seed}.pth'))
    for attention in ('none', 'cbam', 'se'):
        for seed in SEEDS:
            items.append(dict(kind='lts', attention=attention, seed=seed,
                              out=str(lts_run(attention, seed))))
    for seed in SEEDS:
        items.append(dict(kind='detector', attention='se', seed=seed,
                          out=str(detector_run('se', seed, False)), tiled=False, reuse_only=True))
    for job in items:
        job['id'] = Path(job['out']).stem
        job['config'] = str(CONFIG / 'train' / (job['id'] + '.yaml'))
    return items


def prepare():
    local = read(DATA / 'data.yaml')
    local['path'] = str((ROOT / DATA).resolve())
    write_config(DATA / 'data_table2_local.yaml', local)
    for job in jobs():
        seed, attention = job['seed'], job['attention']
        if job['kind'] == 'detector':
            cfg = read('configs/train/detector_kayuan.yaml')
            cfg['dataset'] = dict(name='kaiyuan_cd', data_yaml=str(DATA / 'data_table2_local.yaml'))
            cfg['tiling'].update(type='uniform' if job['tiled'] else 'none', suffix='_lat640_ov20_v1')
            cfg['model'].update(arch='yolo26m', attention=attention, pretrained=None)
            cfg['train'].update(seeds=[seed], device=0, epochs=100)
            cfg['output'] = dict(project=str(RUNS), name=job['id'].rsplit('_seed', 1)[0])
        elif job['kind'] == 'selector':
            cfg = read(f'weights/lat_yolo/kaiyuan_cd/tilecnn_seed{seed}.yaml')
            cfg.pop('gridnet', None)
        else:
            cfg = read('configs/train/lts_kayuan.yaml')
            cfg['training_mode'] = 'joint'
            cfg['out'] = job['out']
            cfg['detector']['weights'] = str(detector_run(attention, seed) / 'weights/best.pt')
            cfg['features']['feature_stage'] = 4 if attention == 'none' else 5
            cfg['train'].update(seed=seed, device='cuda:0', epochs=50, pos_weight=1, lambda_rel=1,
                                amp_dtype='bfloat16')
        write_config(job['config'], cfg)
    for row, tiling, selector, attention in ROWS:
        for seed in SEEDS:
            cfg = read('configs/eval/lts_kayuan.yaml')
            cfg['dataset'] = dict(name='kaiyuan_cd', data_yaml=str(DATA / 'data.yaml'),
                images=str(DATA / 'test/images'), labels=str(DATA / 'test/labels'),
                split_manifest=str(DATA / 'split_manifest.json'))
            cfg['tiling']['type'] = tiling
            cfg['tile_selector'] = {'type': [selector], **(
                {selector: copy.deepcopy(cfg['tile_selector'][selector])} if selector != 'none' else {})}
            cfg['detector']['device'] = 'cuda:0'
            cfg['detector']['weights'] = str(detector_run(attention, seed, tiling != 'none') / 'weights/best.pt')
            if selector == 'tile_cnn':
                cfg['tile_selector'][selector].update(weights=f'weights/lat_yolo/kaiyuan_cd/tilecnn_seed{seed}.pth', device='cuda:0')
            if selector == 'lts':
                cfg['detector']['weights'] = '???'
                cfg['tile_selector'][selector]['feature_stage'] = 4 if attention == 'none' else 5
            cfg['eval'].update(experiment_id=f'kaiyuan_cd_table2_{row}', seed=seed,
                profile_stages=False, visualize=0,
                out=f'outputs/lat_yolo/kaiyuan_cd/table2_test/{row}_seed{seed}.json')
            write_config(CONFIG / 'eval_templates' / f'{row}_seed{seed}.yaml', cfg)
    write_json(STATE / 'plan.json', dict(seeds=SEEDS, physical_gpu=1, gpu_uuid=GPU_UUID,
               detector_epochs=100, lts_epochs=50, jobs=jobs(), table_rows=ROWS))


def verify(job):
    """Validate real artifacts, not just queue markers. Returns completion evidence."""
    import torch
    from adaptile.detectors import register_custom_modules
    from adaptile.provenance import file_sha256
    register_custom_modules()
    cfg = read(job['config'])
    out = Path(job['out'])
    if not out.exists():
        return None
    if job['kind'] == 'selector':
        from adaptile.models.tile_cnn import TileProposalCNN, load_state_dict
        sidecar = out.with_suffix('.yaml')
        if not sidecar.exists():
            return None
        assert read(sidecar)['tile_cnn'] == cfg['tile_cnn'], f'CNN recipe mismatch: {out}'
        load_state_dict(TileProposalCNN(), out, 'cpu')
        return dict(weights=str(out), sha256=file_sha256(out))
    if job['kind'] == 'detector':
        weights = out / 'weights/best.pt'
        if not all(p.exists() for p in (weights, out / 'args.yaml', out / 'results.csv')):
            return None
        args = read(out / 'args.yaml')
        for key, value in dict(seed=job['seed'], epochs=100, imgsz=640, batch=16,
                               patience=30, deterministic=True, optimizer='auto').items():
            assert args[key] == value, f'{out}: {key} mismatch'
        if (out / 'train_config.yaml').exists():
            recorded = read(out / 'train_config.yaml')
            assert recorded['model'] == cfg['model'] and recorded['tiling']['type'] == cfg['tiling']['type'], out
            if job['tiled']:
                for key in ('tile_size', 'overlap', 'min_visibility', 'suffix'):
                    assert recorded['tiling'][key] == cfg['tiling'][key], out
        # Some archived runs lack the wrapper sidecar. Verify the actual model,
        # Ultralytics arguments and completed history instead of manufacturing it.
        from adaptile.training import ARCHS
        architecture = ARCHS[('yolo26m', job['attention'])]
        assert args['model'] == architecture or (
            args.get('resume') == str(out / 'weights/last.pt')
            and args['model'] == args['resume']), out
        accepted_data = {'data_lat640_ov20_v1.yaml'} if job['tiled'] else {'data.yaml', 'data_table2_local.yaml'}
        assert Path(args['data']).name in accepted_data, out
        assert 'kaiyuan_cd' in args['data'], out
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        assert ckpt['model'].names == read(DATA / 'data.yaml')['names'], out
        definition = read(architecture)
        for key in ('backbone', 'head', 'end2end', 'reg_max', 'scales'):
            assert ckpt['model'].yaml[key] == definition[key], f'{out}: architecture mismatch ({key})'
        if ckpt['epoch'] != -1 or ckpt.get('optimizer') is not None:
            return None
        history = list(csv.DictReader((out / 'results.csv').open()))
        assert history and all(math.isfinite(float(v)) for r in history for v in r.values()), out
        if not (out / 'train_config.yaml').exists():
            assert len(history) == 100, f'Archived run lacks completion evidence: {out}'
        return dict(weights=str(weights), epochs=len(history), sha256=file_sha256(weights))
    resume = out / 'resume.pt'
    if not resume.exists():
        return None
    state = torch.load(resume, map_location='cpu', weights_only=False)
    assert state['metadata']['config'] == cfg, f'LTS config mismatch: {out}'
    if state['epoch'] + 1 < cfg['train']['epochs'] and state['bad_epochs'] < cfg['train']['patience']:
        return None
    history = json.loads((out / 'history.json').read_text())
    assert history == state['history'] and len(history) == state['epoch'] + 1, out
    assert all(math.isfinite(v) for r in history for split in ('train', 'val') for v in r[split].values()), out
    assert any(r['train']['det_loss'] > 0 for r in history), f'Not a joint training run: {out}'
    best = json.loads((out / 'best.json').read_text())
    pair = out / best['directory']
    from ultralytics import YOLO
    from adaptile.models.coarse_features import YOLOCoarseFeatureProvider
    from adaptile.models.lts_checkpoint import load_lts_checkpoint
    model = YOLO(str(pair / 'detector.pt')).model.float().cpu()
    provider = YOLOCoarseFeatureProvider(model, **cfg['features'])
    _, meta = load_lts_checkpoint(pair / 'lts.pt', provider, pair / 'detector.pt')
    assert meta['training_mode'] == 'joint' and meta['training']['seed'] == job['seed'], out
    assert meta['training']['source_detector_sha256'] == file_sha256(cfg['detector']['weights']), out
    assert meta['training']['split_manifest_sha256'] == file_sha256(cfg['dataset']['split_manifest']), out
    assert meta['training']['data_yaml_sha256'] == file_sha256(cfg['dataset']['data_yaml']), out
    return dict(pair=str(pair), epochs=len(history), best_epoch=best['epoch'],
                detector_sha256=file_sha256(pair / 'detector.pt'), lts_sha256=file_sha256(pair / 'lts.pt'))


def audit():
    evidence = {}
    for job in jobs():
        evidence[job['id']] = verify(job)
    rows = []
    for row, tiling, selector, attention in ROWS:
        for seed in SEEDS:
            dependencies = [detector_run(attention, seed, tiling != 'none').name]
            if selector == 'tile_cnn':
                dependencies.append(f'tilecnn_seed{seed}')
            if selector == 'lts':
                dependencies.append(lts_run(attention, seed).name)
            ready = all(evidence[d] is not None for d in dependencies)
            cfg_path = CONFIG / 'eval_templates' / f'{row}_seed{seed}.yaml'
            if ready:
                cfg = read(cfg_path)
                if selector == 'lts':
                    pair = evidence[dependencies[-1]]['pair']
                    cfg['detector']['weights'] = f'{pair}/detector.pt'
                    cfg['tile_selector']['lts']['weights'] = f'{pair}/lts.pt'
                cfg_path = CONFIG / 'eval' / cfg_path.name
                write_config(cfg_path, cfg)
            rows.append(dict(row=row, seed=seed, train_complete=ready,
                             dependencies=dependencies, eval_config=str(cfg_path)))
    report = dict(updated_at=datetime.now(timezone.utc).isoformat(), jobs=evidence, rows=rows,
                  complete=sum(r['train_complete'] for r in rows), total=len(rows), test_executed=False)
    write_json(STATE / 'audit.json', report)
    print(f'Train-ready Table II cells: {report["complete"]}/{report["total"]}', flush=True)
    return report


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('Use scripts/run_kaiyuan_table2.sh: physical GPU 1 must be isolated by UUID')
    if Path(sys.prefix).resolve() != Path('/venv/LAT-YOLO').resolve():
        raise RuntimeError('Activate conda LAT-YOLO first')
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in jobs():
            done = verify(job)
            if done:
                print(f'SKIP verified {job["id"]}', flush=True)
                continue
            if job.get('reuse_only'):
                raise RuntimeError(f'Baseline must be reused, never retrained: {job["out"]}')
            entry = {'detector': 'train_detector.py', 'selector': 'train_selector.py', 'lts': 'train_lts.py'}[job['kind']]
            command = [sys.executable, '-u', f'scripts/{entry}', job['config']]
            if Path(job['out']).exists():
                resume = Path(job['out']) / 'resume.pt'
                if job['kind'] == 'lts' and resume.is_file():
                    command.extend(['--resume', str(resume)])
                else:
                    raise RuntimeError(f'Incomplete existing output requires inspection: {job["out"]}')
            log_path = STATE / f'{job["id"]}.log'
            print(f'START {job["id"]} log={log_path}', flush=True)
            with log_path.open('a', buffering=1) as log:
                log.write('\nCOMMAND ' + json.dumps(command) + '\n')
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                write_json(STATE / 'current.json', dict(job=job, pid=process.pid, command=command,
                    log=str(log_path), status='running', started_at=datetime.now(timezone.utc).isoformat()))
                code = process.wait()
            write_json(STATE / 'current.json', dict(job=job, pid=process.pid, exit_code=code,
                                                   log=str(log_path), status='exited'))
            if code:
                raise RuntimeError(f'{job["id"]} failed ({code}); inspect {log_path}')
            if verify(job) is None:
                raise RuntimeError(f'Process exited successfully but training is incomplete: {job["id"]}')
            audit()
            print(f'COMPLETE {job["id"]}', flush=True)
        report = audit()
        assert report['complete'] == 27
        print('ALL 27 TABLE II TRAINING CELLS READY. Test evaluation remains manual.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'audit', 'run'))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'prepare':
        prepare()
    elif args.action == 'audit':
        import torch
        torch.set_num_threads(4)
        audit()
    else:
        run()


if __name__ == '__main__':
    main()
