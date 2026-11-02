#!/usr/bin/env bash
# Three seeds, each detector -> joint LTS -> full validation threshold sweep.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LAT_PYTHON="${LAT_PYTHON:-/opt/anaconda3/envs/traffic_light/bin/python}"
LAT_RUN_ROOT="${1:-runs/lat_yolo/visdrone/se_3seeds_$(date +%Y%m%d_%H%M%S)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export WANDB_MODE=disabled

# Refuse to overwrite a previous experiment. Config snapshots use local cuda:0,
# which maps exclusively to physical GPU 1 through CUDA_VISIBLE_DEVICES above.
"$LAT_PYTHON" - "$LAT_RUN_ROOT" <<'PY'
import sys
import json
from pathlib import Path
from omegaconf import OmegaConf
from adaptile.provenance import source_fingerprints, software_versions, file_sha256
root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
(root / 'configs').mkdir()
(root / 'validation').mkdir()
det = OmegaConf.load('configs/train/detector_visdrone.yaml')
det.output.project = str(root)
det.train.device = 0
assert list(det.train.seeds) == [0, 42, 100]
lts = OmegaConf.load('configs/train/lts_visdrone.yaml')
lts.train.device = 'cuda:0'
assert lts.train.epochs == 100
for seed in [0, 42, 100]:
    det.train.seeds = [seed]
    OmegaConf.save(det, root / 'configs' / f'detector_seed{seed}.yaml')
    lts.train.seed = seed
    lts.out = str(root / f'lts_joint_se_seed{seed}')
    lts.detector.weights = str(root / f'yolo26m_p2_tiled_se_seed{seed}/weights/best.pt')
    OmegaConf.save(lts, root / 'configs' / f'lts_seed{seed}.yaml')
ev = OmegaConf.load('configs/eval/lts_visdrone.yaml')
ev.detector.device = 'cuda:0'
OmegaConf.save(ev, root / 'configs' / 'validation_template.yaml')
plan = {'seeds': [0, 42, 100], 'validation_thresholds': [0.3, 0.5],
        'physical_gpu': 1, 'detector_batch': det.train.batch,
        'lts_batch': lts.train.batch_size, 'lts_epochs': lts.train.epochs,
        'lts_patience': lts.train.patience,
        'workers': {'detector': det.train.workers, 'lts': lts.train.workers},
        'source_sha256': source_fingerprints(), 'software': software_versions(),
        'split_manifest_sha256': file_sha256(lts.dataset.split_manifest)}
(root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
PY

echo "$$" > "$LAT_RUN_ROOT/pipeline.pid"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "failed exit=$rc at $(date -Is)" > "$LAT_RUN_ROOT/status.txt"; fi' EXIT
for LAT_SEED in 0 42 100; do
    echo "detector seed=$LAT_SEED started $(date -Is)" > "$LAT_RUN_ROOT/status.txt"
    "$LAT_PYTHON" -u scripts/train_detector.py "$LAT_RUN_ROOT/configs/detector_seed${LAT_SEED}.yaml" > "$LAT_RUN_ROOT/detector_seed${LAT_SEED}.log" 2>&1
    echo "joint LTS seed=$LAT_SEED started $(date -Is)" > "$LAT_RUN_ROOT/status.txt"
    "$LAT_PYTHON" -u scripts/train_lts.py "$LAT_RUN_ROOT/configs/lts_seed${LAT_SEED}.yaml" > "$LAT_RUN_ROOT/lts_seed${LAT_SEED}.log" 2>&1

    # Resolve the immutable matched pair only after joint training finishes.
    "$LAT_PYTHON" - "$LAT_RUN_ROOT" "$LAT_SEED" <<'PY'
import json
import math
import sys
from pathlib import Path
from omegaconf import OmegaConf
root, seed = Path(sys.argv[1]).resolve(), int(sys.argv[2])
run = root / f'lts_joint_se_seed{seed}'
history = json.loads((run / 'history.json').read_text())
training = OmegaConf.load(root / 'configs' / f'lts_seed{seed}.yaml')
assert history and [row['epoch'] for row in history] == list(range(1, len(history) + 1))
assert len(history) <= training.train.epochs == 100
assert all(math.isfinite(row[split][key]) for row in history
           for split in ('train', 'val') for key in ('loss', 'det_loss', 'rel_loss'))
best = json.loads((run / 'best.json').read_text())
assert len(history) == training.train.epochs or len(history) - best['epoch'] >= training.train.patience, 'LTS stopped before its configured schedule or early stopping condition'
pair = run / best['directory']
assert all((pair / name).is_file() for name in ('detector.pt', 'lts.pt', 'metadata.json'))
for threshold, tag in ((0.3, '03'), (0.5, '05')):
    cfg = OmegaConf.load(root / 'configs' / 'validation_template.yaml')
    cfg.detector.weights = str(pair / 'detector.pt')
    cfg.tile_selector.lts.weights = str(pair / 'lts.pt')
    cfg.tile_selector.lts.threshold = threshold
    cfg.eval.seed = seed
    cfg.eval.experiment_id = f'visdrone_joint_se_validation_thr{tag}'
    cfg.eval.out = str(root / 'validation' / f'val_seed{seed}_thr{tag}.json')
    OmegaConf.save(cfg, root / 'configs' / f'val_seed{seed}_thr{tag}.yaml')
PY
    for LAT_THRESHOLD_TAG in 03 05; do
        echo "validation seed=$LAT_SEED threshold=0.${LAT_THRESHOLD_TAG#0} started $(date -Is)" > "$LAT_RUN_ROOT/status.txt"
        "$LAT_PYTHON" -u scripts/benchmark_lts.py "$LAT_RUN_ROOT/configs/val_seed${LAT_SEED}_thr${LAT_THRESHOLD_TAG}.yaml" > "$LAT_RUN_ROOT/val_seed${LAT_SEED}_thr${LAT_THRESHOLD_TAG}.log" 2>&1
    done
    echo "seed=$LAT_SEED detector+LTS+sweep completed $(date -Is)" >> "$LAT_RUN_ROOT/completed_seeds.txt"
done

echo "aggregating validation sweep $(date -Is)" > "$LAT_RUN_ROOT/status.txt"
for LAT_THRESHOLD_TAG in 03 05; do
    "$LAT_PYTHON" scripts/collect_e02.py \
        "$LAT_RUN_ROOT/validation/val_seed0_thr${LAT_THRESHOLD_TAG}.json" \
        "$LAT_RUN_ROOT/validation/val_seed42_thr${LAT_THRESHOLD_TAG}.json" \
        "$LAT_RUN_ROOT/validation/val_seed100_thr${LAT_THRESHOLD_TAG}.json" \
        --out "$LAT_RUN_ROOT/validation/summary_thr${LAT_THRESHOLD_TAG}.json"
done
echo "completed $(date -Is)" > "$LAT_RUN_ROOT/status.txt"
