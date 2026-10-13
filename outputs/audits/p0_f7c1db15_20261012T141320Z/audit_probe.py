"""Read-only P0 probes; write evidence exclusively into a new --out directory.

Run with CUDA_VISIBLE_DEVICES='' and the existing traffic_light Python.
No detector training, dataset mutation, network access, or LTS implementation.
"""

import argparse
import contextlib
import csv
import datetime as dt
import gc
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def command(*args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()}


def stamp(path):
    path = Path(path)
    stat = path.stat()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest,
            'bytes': stat.st_size,
            'mtime_utc': dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat()}


def label_directory(images):
    parts = list(images.parts)
    parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
    return Path(*parts)


def dataset_inventory(root):
    config = yaml.safe_load((root / 'data.yaml').read_text())
    base = Path(config.get('path', root))
    if not base.is_absolute():
        base = root / base
    result = {'yaml': stamp(root / 'data.yaml'), 'config': config, 'splits': {},
              'license_status': 'not_verified', 'near_duplicate_audit': 'not_run',
              'image_content_hashes': 'not_computed_in_P0'}
    manifest_path = root / 'split_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        groups = {s: {x['sequence'] for x in manifest['samples'] if x['split'] == s}
                  for s in ('train', 'val', 'test')}
        result['manifest'] = {**stamp(manifest_path), 'seed': manifest['seed'],
            'sample_count': len(manifest['samples']),
            'group_counts': {s: len(g) for s, g in groups.items()},
            'group_intersections': {a + '/' + b: sorted(groups[a] & groups[b])
                                   for a, b in itertools.combinations(groups, 2)}}
        mismatches = []
        for row in manifest['samples']:
            local = base / config[row['split']] / Path(row['image']).name
            if local.resolve() != Path(row['image']).resolve():
                mismatches.append(str(local))
        result['manifest']['destination_target_mismatches'] = mismatches
    for split in ('train', 'val', 'test'):
        if split not in config:
            continue
        images = base / config[split]
        labels = label_directory(images)
        ipaths = sorted(p for p in images.glob('*') if p.suffix.lower() in EXTENSIONS)
        lpaths = sorted(labels.glob('*.txt'))
        ibroken = [p for p in ipaths if not p.is_file()]
        lbroken = [p for p in lpaths if not p.is_file()]
        missing = [p.name for p in ipaths if not (labels / (p.stem + '.txt')).is_file()]
        keys = [p.name for p in ipaths]
        result['splits'][split] = {
            'images_path': str(images.relative_to(ROOT)),
            'labels_path': str(labels.relative_to(ROOT)),
            'image_entries': len(ipaths), 'label_entries': len(lpaths),
            'readable_image_files': len(ipaths) - len(ibroken),
            'broken_image_files': len(ibroken), 'broken_label_files': len(lbroken),
            'missing_label_count': len(missing), 'missing_label_examples': missing[:5],
            'broken_target_examples': [str(p.resolve()) for p in ibroken[:3]],
            'sorted_image_names_sha256': hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
            'duplicate_image_stems': len(ipaths) - len({p.stem for p in ipaths}),
            'label_validation': 'not_run_in_P0'}
    return result


def checkpoint_inventory(path):
    entry = stamp(path)
    # These are existing local experiment checkpoints, not downloaded artifacts.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    entry['top_level_keys'] = list(ckpt) if isinstance(ckpt, dict) else None
    if path.suffix == '.pt':
        model = ckpt.get('ema') if ckpt.get('ema') is not None else ckpt.get('model')
        if model is not None:
            entry.update(model_class=type(model).__name__, names=getattr(model, 'names', None),
                nc=getattr(model.model[-1], 'nc', None),
                architecture=getattr(model, 'yaml', None),
                parameters=sum(p.numel() for p in model.parameters()),
                strides=model.stride.tolist() if hasattr(model, 'stride') else None,
                attention_modules=[{'name': name, 'type': type(mod).__name__}
                    for name, mod in model.named_modules() if type(mod).__name__ in ('SE', 'CBAM')])
        entry['checkpoint_metadata'] = {k: ckpt.get(k) for k in
            ('date', 'version', 'epoch', 'train_args', 'train_metrics')}
    else:
        entry['selector_metadata'] = {k: v for k, v in ckpt.items()
            if not isinstance(v, (torch.Tensor, dict))}
        state = ckpt.get('state', ckpt.get('state_dict', ckpt))
        entry['tensor_shapes'] = {k: list(v.shape) for k, v in state.items() if torch.is_tensor(v)}
        entry['training_provenance'] = 'historical; clean-split eligibility not established'
    del ckpt
    gc.collect()
    return entry


def architecture_probes():
    from adaptile.detectors import register_custom_modules
    from adaptile.training import ARCHS
    from ultralytics import YOLO
    register_custom_modules()
    rows = []
    for (arch, attention), definition in ARCHS.items():
        row = {'arch': arch, 'attention': attention, 'definition': definition,
               'device': 'cpu', 'input_shape': [1, 3, 64, 64],
               'optimizer_step': False, 'training_loss_verified': False}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(0)
                model = YOLO(definition).model.eval()
            before = {id(p) for p in model.parameters()}
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            with torch.no_grad():
                model(torch.zeros(1, 3, 64, 64))
            registered = {id(p) for group in optimizer.param_groups for p in group['params']}
            row.update(status='verified_cpu_forward', strides=model.stride.tolist(),
                detect_from=model.model[-1].f, end2end=getattr(model, 'end2end', False),
                nc=model.model[-1].nc, parameters=sum(p.numel() for p in model.parameters()),
                parameters_added_by_explicit_forward=len({id(p) for p in model.parameters()} - before),
                parameters_missing_from_optimizer=[n for n, p in model.named_parameters() if id(p) not in registered],
                attention_modules=[{'name': n, 'type': type(m).__name__,
                    'parameters': sum(p.numel() for p in m.parameters())}
                    for n, m in model.named_modules() if type(m).__name__ in ('CBAM', 'SE')])
            del model, optimizer
        except Exception as exc:
            row.update(status='failed', error=repr(exc))
        rows.append(row)
        gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pytest-xml', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / 'baseline_inventory.json'
    if target.exists():
        raise FileExistsError(target)
    os.chdir(ROOT)
    torch.set_num_threads(2)
    from adaptile.detectors import register_custom_modules
    register_custom_modules()
    baseline = {
        'status': 'verified_P0_snapshot_with_blockers; not_research_ready',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'root': str(ROOT),
        'git': {key: command(*cmd) for key, cmd in {
            'sha': ('git', 'rev-parse', 'HEAD'),
            'branch': ('git', 'branch', '--show-current'),
            'status': ('git', 'status', '--short', '--branch'),
            'diff': ('git', 'diff', '--', 'configs/train/detector_kayuan.yaml'),
            'branches': ('git', 'branch', '-avv'),
            'ignored_data_tools': ('git', 'check-ignore', '-v',
                'scripts/data/check_split_leakage.py', 'scripts/data/convert_visdrone_to_yolo.py',
                'scripts/data/prepare_bstld.py')}.items()},
        'environment': {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'torch_cuda_build': torch.version.cuda,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'packages': {p: importlib.metadata.version(p) for p in (
                'torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python',
                'omegaconf', 'pytest', 'PyYAML', 'ultralytics-thop')},
            'gpu_snapshot': command('nvidia-smi',
                '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv'),
            'cpu': command('lscpu')},
        'limitations': ['No manuscript PDF found in repository or attachment search.',
            'No training, GPU inference, speed benchmark, downloads or branch switching performed.',
            'No duplicate/near-duplicate image or full annotation validation performed.',
            'Checkpoint hashes and metadata do not prove training sample membership.',
            'Filesystem mtimes are not original experiment timestamps.',
            'Dataset licenses and official evaluation protocols not verified.'],
    }
    sources = set()
    for folder in ('src', 'scripts', 'test', 'configs'):
        sources.update(p for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.yaml', '.sh'))
    sources.update(ROOT / p for p in ('spec.md', 'implementation_plan.md', 'README.md', 'RUNBOOK.md',
        'HANDOFF.md', 'requirements.txt', 'pyproject.toml', '.gitignore', 'weights/README.md', 'outputs/README.md'))
    baseline['source_files'] = [stamp(p) for p in sorted(sources)]
    baseline['configs'] = {str(p.relative_to(ROOT)): yaml.safe_load(p.read_text())
                           for p in sorted((ROOT / 'configs').rglob('*.yaml'))}
    baseline['datasets'] = {str(p.parent.relative_to(ROOT)): dataset_inventory(p.parent)
        for p in sorted((ROOT / 'data').glob('*/data.yaml'))}
    baseline['bstld'] = {'yaml_exists': (ROOT / 'data/BSTLD/data.yaml').exists(),
        'archive': command('file', 'data/BSTLD/bstld-DatasetNinja.tar'),
        'status': 'archive_present; extraction/conversion/license not_verified'}
    baseline['architecture_probes'] = architecture_probes()
    checkpoints = sorted(p for folder in ('weights', 'runs') for p in (ROOT / folder).rglob('*')
                         if p.suffix in ('.pt', '.pth'))
    baseline['checkpoints'] = []
    for path in checkpoints:
        try:
            entry = checkpoint_inventory(path)
        except Exception as exc:
            entry = {**stamp(path), 'metadata_error': repr(exc)}
        baseline['checkpoints'].append(entry)
    baseline['historical_outputs'] = []
    for path in sorted((ROOT / 'outputs').rglob('*')):
        if 'audits' in path.parts or path.suffix not in ('.json', '.csv', '.yaml', '.md'):
            continue
        entry = stamp(path)
        if path.suffix == '.json':
            entry['saved_results'] = json.loads(path.read_text())
        entry['status'] = 'historical; not_new_LAT_YOLO_results'
        baseline['historical_outputs'].append(entry)
    baseline['training_runs'] = []
    for path in sorted((ROOT / 'runs').rglob('args.yaml')):
        entry = {**stamp(path), 'args': yaml.safe_load(path.read_text())}
        for name in ('train_config.yaml', 'results.csv'):
            file = path.parent / name
            if file.exists():
                entry[name] = stamp(file)
                if name.endswith('.yaml'):
                    entry[name]['config'] = yaml.safe_load(file.read_text())
                else:
                    rows = list(csv.DictReader(io.StringIO(file.read_text())))
                    entry[name].update(epochs_logged=len(rows), last_row=rows[-1] if rows else None)
        baseline['training_runs'].append(entry)
    if args.pytest_xml:
        shutil.copyfile(args.pytest_xml, args.out / 'baseline_tests.xml')
    with target.open('x') as handle:
        json.dump(baseline, handle, indent=2, default=str)
        handle.write('\n')
    print(json.dumps({'inventory': str(target), 'checkpoints': len(checkpoints),
                      'architectures': baseline['architecture_probes']}, indent=2))


if __name__ == '__main__':
    main()
