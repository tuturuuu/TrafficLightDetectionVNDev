#!/usr/bin/env python
"""Split an existing VisDrone experiment across GPUs without restarting its detector.

Kept below scripts/runtime so queue management does not change research source
fingerprints (which include scripts/*.py and src/adaptile/**/*.py).
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def lock(path, nonblocking=False):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def process_info(pid):
    path = Path('/proc') / str(pid)
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]), 'raw_exit_status': int(fields[49]),
            'command': [v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v]}


def refresh_overview(root):
    states = [json.loads(p.read_text()) for p in sorted(root.glob('queue_gpu*.json'))]
    lines = [f"GPU {s['gpu']}: {s['stage']} seed={s.get('seed')} pid={s['pid']} updated={s['updated_at']}"
             + (f" error={s['error']}" if s.get('error') else '') for s in states]
    (root / 'status.txt').write_text('\n'.join(lines) + '\n')


def state(root, gpu, stage, **details):
    with lock(root / '.queue_state.lock'):
        atomic_json(root / f'queue_gpu{gpu}.json',
                    {'gpu': gpu, 'pid': os.getpid(), 'stage': stage, 'updated_at': now(), **details})
        refresh_overview(root)


def adopt_detector(root, gpu, manifest_path, poll_seconds=15):
    """Read the original child's wait status while its original parent is stopped.

    The adopted process is not our child, so waitpid/psutil.wait cannot obtain
    its exit code. A stopped original parent leaves it as a zombie until we
    read Linux stat field 52. Never infer success from a missing process.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    expected_parent, expected_child = manifest['parent'], manifest['child']

    def parent_info():
        parent = process_info(expected_parent['pid'])
        if (parent['start_ticks'] != expected_parent['start_ticks']
                or parent['command'] != expected_parent['command'] or parent['state'] != 'T'):
            raise RuntimeError('Original controller identity/stopped state changed')
        return parent

    parent_info()
    child = process_info(expected_child['pid'])
    if (child['start_ticks'] != expected_child['start_ticks']
            or child['ppid'] != expected_parent['pid']
            or (child['state'] != 'Z' and child['command'] != expected_child['command'])):
        raise RuntimeError('Adopted detector identity changed')
    state(root, gpu, 'detector (adopted)', seed=0, child_pid=child['pid'])
    atomic_json(root / 'handoff_ready.json', {'worker_pid': os.getpid(), 'created_at': now(),
                                            'manifest': str(manifest_path)})
    while True:
        parent_info()
        child = process_info(expected_child['pid'])
        if child['start_ticks'] != expected_child['start_ticks'] or child['ppid'] != expected_parent['pid']:
            raise RuntimeError('Adopted detector PID identity changed while waiting')
        if child['state'] == 'Z':
            raw = child['raw_exit_status']
            code = os.waitstatus_to_exitcode(raw)
            atomic_json(root / 'adopted_detector_exit.json',
                        {'detector': child, 'exit_code': code, 'observed_at': now()})
            parent_info()
            os.kill(expected_parent['pid'], signal.SIGKILL)  # Only controller; never its process group.
            if code:
                raise RuntimeError(f'Adopted detector failed with exit code {code}')
            return
        time.sleep(poll_seconds)


def run_stage(root, gpu, seed, stage, script, config, log_name):
    log = root / log_name
    with log.open('x') as output:
        state(root, gpu, stage, seed=seed, log=str(log))
        child = subprocess.Popen([sys.executable, '-u', script, str(config)], cwd=REPO,
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        state(root, gpu, stage, seed=seed, child_pid=child.pid, log=str(log))
        code = child.wait()
    if code:
        raise RuntimeError(f'{stage} seed {seed} exited {code}; see {log}')


def validate_detector(root, seed):
    import csv
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(root / 'configs' / f'detector_seed{seed}.yaml')
    run = root / f'yolo26m_p2_tiled_se_seed{seed}'
    rows = list(csv.DictReader((run / 'results.csv').open()))
    if not rows or [int(row['epoch']) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Detector history is empty or noncontiguous')
    if len(rows) > cfg.train.epochs or not all(math.isfinite(float(v)) for row in rows for v in row.values()):
        raise ValueError('Detector history exceeds its schedule or contains nonfinite values')
    if not (run / 'weights' / 'best.pt').is_file():
        raise FileNotFoundError('Detector best checkpoint is missing')


def prepare_validation(root, seed):
    from omegaconf import OmegaConf
    run = root / f'lts_joint_se_seed{seed}'
    history = json.loads((run / 'history.json').read_text())
    training = OmegaConf.load(root / 'configs' / f'lts_seed{seed}.yaml')
    if not history or [r['epoch'] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError('LTS history is empty or noncontiguous')
    if not len(history) <= training.train.epochs == 100:
        raise ValueError('Unexpected LTS training schedule')
    if not all(math.isfinite(r[s][k]) for r in history for s in ('train', 'val')
               for k in ('loss', 'det_loss', 'rel_loss')):
        raise ValueError('Nonfinite joint loss')
    best = json.loads((run / 'best.json').read_text())
    if len(history) != training.train.epochs and len(history) - best['epoch'] < training.train.patience:
        raise ValueError('LTS ended before its schedule/early stopping condition')
    pair = run / best['directory']
    if not all((pair / name).is_file() for name in ('detector.pt', 'lts.pt', 'metadata.json')):
        raise FileNotFoundError('Matched detector/LTS pair is incomplete')
    for threshold, tag in ((0.3, '03'), (0.5, '05')):
        cfg = OmegaConf.load(root / 'configs' / 'validation_template.yaml')
        cfg.detector.weights = str(pair / 'detector.pt')
        cfg.tile_selector.lts.weights = str(pair / 'lts.pt')
        cfg.tile_selector.lts.threshold = threshold
        cfg.eval.seed = seed
        cfg.eval.experiment_id = f'visdrone_joint_se_validation_thr{tag}'
        cfg.eval.out = str(root / 'validation' / f'val_seed{seed}_thr{tag}.json')
        destination = root / 'configs' / f'val_seed{seed}_thr{tag}.yaml'
        if destination.exists():
            raise FileExistsError(destination)
        OmegaConf.save(cfg, destination)


def validate_sweep(root, seed):
    from omegaconf import OmegaConf
    template = OmegaConf.load(root / 'configs' / 'validation_template.yaml')
    expected_images = sum(p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                          for p in Path(template.dataset.images).iterdir())
    for threshold, tag in ((0.3, '03'), (0.5, '05')):
        path = root / 'validation' / f'val_seed{seed}_thr{tag}.json'
        rows = json.loads(path.read_text())
        if len(rows) != 2 or {r['method'] for r in rows} != {'uniform', 'lts'}:
            raise ValueError(f'Incomplete selector comparisons: {path}')
        for row in rows:
            if row['n_images'] != expected_images or row['provenance']['seed'] != seed:
                raise ValueError(f'Wrong validation coverage or seed: {path}')
            if row['schema_version'] != 'lat_yolo_v1':
                raise ValueError(f'Wrong evaluation protocol: {path}')
            if row['method'] == 'lts' and row['provenance']['selector']['threshold'] != threshold:
                raise ValueError(f'Wrong selector threshold: {path}')
            if not all(math.isfinite(row[k]) for k in ('mAP50', 'mAP50-95', 'precision', 'recall', 'f1', 'fps')):
                raise ValueError(f'Nonfinite validation metrics: {path}')
        if not all(path.with_suffix(suffix).is_file() for suffix in ('.yaml', '.predictions.json')):
            raise FileNotFoundError(f'Missing validation sidecars: {path}')


def mark_complete(root, seed, gpu):
    with lock(root / '.queue_state.lock'):
        marker = root / f'seed{seed}.completed.json'
        if marker.exists():
            raise FileExistsError(marker)
        atomic_json(marker, {'seed': seed, 'gpu': gpu, 'completed_at': now(),
                             'thresholds': [0.3, 0.5]})
        with (root / 'completed_seeds.txt').open('a') as handle:
            handle.write(f'seed={seed} detector+LTS+sweep completed {now()} physical_gpu={gpu}\n')


def maybe_aggregate(root):
    with lock(root / '.aggregate.lock'):
        if not all((root / f'seed{s}.completed.json').exists() for s in (0, 42, 100)):
            return False
        if (root / 'all_seeds.completed.json').exists():
            if not all((root / 'validation' / f'summary_thr{tag}{suffix}').is_file()
                       for tag in ('03', '05') for suffix in ('.json', '.csv')):
                raise FileNotFoundError('Completion marker exists but summary artifacts are missing')
            return True
        for seed in (0, 42, 100):
            validate_sweep(root, seed)
        for tag in ('03', '05'):
            output = root / 'validation' / f'summary_thr{tag}.json'
            if output.exists() or output.with_suffix('.csv').exists():
                raise FileExistsError(f'Report already exists; inspect before repeating: {output}')
            paths = [str(root / 'validation' / f'val_seed{s}_thr{tag}.json') for s in (0, 42, 100)]
            with (root / 'aggregation.log').open('a') as log:
                subprocess.run([sys.executable, 'scripts/collect_e02.py', *paths, '--out', str(output)],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json(root / 'all_seeds.completed.json', {'completed_at': now(), 'seeds': [0, 42, 100]})
        return True


def worker(root, gpu, seeds, adoption=None):
    os.chdir(REPO)
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                      PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      MPLBACKEND='Agg', WANDB_MODE='disabled')
    root = Path(root).resolve()
    if adoption and (gpu != 1 or seeds != [0, 42]):
        raise ValueError('Adoption is restricted to the existing GPU 1 queue [0, 42]')
    with lock(root / f'.queue_gpu{gpu}.lock', nonblocking=True):
        try:
            state(root, gpu, 'starting', seeds=seeds)
            for seed in seeds:
                with lock(root / f'.seed{seed}.lock', nonblocking=True):
                    if (root / f'seed{seed}.completed.json').exists():
                        raise FileExistsError(f'Seed {seed} is already complete')
                    if adoption and seed == 0:
                        adopt_detector(root, gpu, adoption)
                    else:
                        run_stage(root, gpu, seed, 'detector', 'scripts/train_detector.py',
                                  root / 'configs' / f'detector_seed{seed}.yaml', f'detector_seed{seed}.log')
                    validate_detector(root, seed)
                    run_stage(root, gpu, seed, 'joint LTS', 'scripts/train_lts.py',
                              root / 'configs' / f'lts_seed{seed}.yaml', f'lts_seed{seed}.log')
                    prepare_validation(root, seed)
                    for tag in ('03', '05'):
                        run_stage(root, gpu, seed, f'validation threshold=0.{tag[1]}', 'scripts/benchmark_lts.py',
                                  root / 'configs' / f'val_seed{seed}_thr{tag}.yaml', f'val_seed{seed}_thr{tag}.log')
                    validate_sweep(root, seed)
                    mark_complete(root, seed, gpu)
            state(root, gpu, 'queue completed', seeds=seeds)
            if maybe_aggregate(root):
                state(root, gpu, 'all seeds and summaries completed', seeds=seeds)
        except BaseException as error:
            state(root, gpu, 'failed', seeds=seeds, error=str(error))
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--seeds', type=int, choices=(0, 42, 100), nargs='+', required=True)
    parser.add_argument('--adopt')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct')
    worker(args.root, args.gpu, args.seeds, args.adopt)


if __name__ == '__main__':
    main()
