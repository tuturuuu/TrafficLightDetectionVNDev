"""Short synthetic CPU workflows, never a research training/accuracy claim."""

import copy
import json

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptile.config import load_config
from adaptile.detectors import register_custom_modules
from adaptile.evaluation_lts import run_evaluation
from adaptile.training_lts import train_lts


@pytest.fixture
def fixture_config(tmp_path):
    from ultralytics import YOLO
    register_custom_modules()
    torch.set_num_threads(2)
    torch.manual_seed(10)
    source = tmp_path / 'source.pt'
    # Keep the production P2-P5 architecture in this smoke test.
    YOLO('configs/models/yolo26m-p2.yaml').save(source)
    cfg = OmegaConf.to_container(OmegaConf.load('configs/train/lts_kayuan.yaml'), resolve=True)
    cfg['out'] = str(tmp_path / 'run')
    cfg['detector'] = {'weights': str(source), 'imgsz': 64}
    cfg['dataset']['data_yaml'] = None
    cfg['features'] = {'feature_stage': 4, 'coarse_imgsz': 64}
    cfg['tiling'] = {'tile_size': 64, 'overlap': .2, 'min_visibility': .3}
    cfg['train'].update(epochs=2, batch_size=2, warmup_steps=1, amp=False)
    rng = np.random.default_rng(9)
    for split in ('train', 'val'):
        images, labels = tmp_path / split / 'images', tmp_path / split / 'labels'
        images.mkdir(parents=True); labels.mkdir()
        for i in range(2):
            cv2.imwrite(str(images / f'{i}.png'), rng.integers(0, 256, (81, 99, 3), np.uint8))
            (labels / f'{i}.txt').write_text('0 0.3 0.3 0.2 0.2\n')
        cfg['dataset'][f'{split}_images'] = str(images)
        cfg['dataset'][f'{split}_labels'] = str(labels)
    manifest = tmp_path / 'split_manifest.json'
    manifest.write_text('{"purpose": "synthetic CPU integration test"}')
    cfg['dataset']['split_manifest'] = str(manifest)
    return cfg


@pytest.mark.parametrize('mode', ['frozen_head_pilot', 'joint'])
def test_epoch_resume_matches_uninterrupted_training(fixture_config, tmp_path, mode):
    cfg = fixture_config
    cfg['training_mode'] = mode
    whole = copy.deepcopy(cfg)
    whole['out'] = str(tmp_path / 'whole')
    expected = train_lts(whole)
    first = train_lts(cfg, stop_after_epochs=1)
    assert len(first) == 1
    history = train_lts(cfg, resume=tmp_path / 'run/resume.pt')
    assert history == expected
    actual = torch.load(tmp_path / 'run/resume.pt', weights_only=False)
    reference = torch.load(tmp_path / 'whole/resume.pt', weights_only=False)
    for key, value in reference['system'].items():
        torch.testing.assert_close(actual['system'][key], value, rtol=0, atol=0)
    assert actual['scheduler'] == reference['scheduler']
    with pytest.raises(FileExistsError):
        train_lts(cfg)


def test_joint_train_export_evaluate_and_profile_parity(fixture_config, tmp_path):
    cfg = fixture_config
    cfg['training_mode'] = 'joint'
    history = train_lts(cfg, stop_after_epochs=1)
    assert history[0]['train']['det_loss'] > 0
    best = json.loads((tmp_path / 'run/best.json').read_text())
    pair = tmp_path / 'run' / best['directory']
    overrides = [f'dataset.images={cfg["dataset"]["val_images"]}',
                 'dataset.data_yaml=null',
                 f'dataset.labels={cfg["dataset"]["val_labels"]}',
                 f'dataset.split_manifest={cfg["dataset"]["split_manifest"]}',
                 f'detector.weights={pair}/detector.pt', f'tile_selector.lts.weights={pair}/lts.pt',
                 'detector.imgsz=64', 'detector.batch=2', 'detector.max_det=5', 'detector.conf=0.000001',
                 'tiling.tile_size=64', 'tile_selector.lts.coarse_imgsz=64',
                 'tile_selector.param_free_avg.coarse_imgsz=64',
                 'tile_selector.lts.force_keep_all=true', 'tile_selector.param_free_avg.force_keep_all=true',
                 'eval.n_warmup=1', 'eval.visualize=1', f'eval.out={tmp_path}/evaluation.json']
    evaluation = load_config('configs/eval/lts_kayuan.yaml', overrides)
    rows = run_evaluation(evaluation)
    records = json.loads((tmp_path / 'evaluation.predictions.json').read_text())
    assert len(rows) == 3
    for row in rows:
        assert row['n_images'] == 2
        assert row['kept_per_img'] == row['tiles_per_img']
        assert row['fps'] == pytest.approx(2 / row['wall_s'])
        assert row['latency_mean_ms'] == pytest.approx(1000 / row['fps'])
        assert row['wall_s'] >= row['select_s'] + row['detect_s'] + row['merge_s']
        assert row['provenance']['source_sha256']
        assert row['provenance']['split_manifest_sha256']
        assert row['provenance']['detector_sha256']
        assert row['coco_protocol']['max_dets'] == 100
    assert rows[-1]['provenance']['lts_checkpoint']['training_mode'] == 'joint'
    times = rows[-1]['stage_latency_ms']
    assert set(times) == {'coarse_preprocess_ms', 'coarse_backbone_ms', 'relevance_predict_ms',
                         'candidate_tile_scoring_ms', 'fine_crop_prepare_ms', 'fine_detector_ms',
                         'remap_ms', 'nms_merge_ms', 'end_to_end_ms'}
    assert sum(v['mean'] for k, v in times.items() if k != 'end_to_end_ms') <= times['end_to_end_ms']['mean']
    for record in records[1:]:
        for uniform, selected in zip(records[0]['images'], record['images']):
            assert uniform['predictions']
            np.testing.assert_allclose(uniform['predictions'], selected['predictions'])
    minimal = load_config('configs/eval/lts_kayuan.yaml', overrides +
                         ['eval.profile_stages=false', 'eval.visualize=0', f'eval.out={tmp_path}/minimal.json'])
    minimal_rows = run_evaluation(minimal)
    minimal_records = json.loads((tmp_path / 'minimal.predictions.json').read_text())
    for instrumented, plain in zip(records, minimal_records):
        for a, b in zip(instrumented['images'], plain['images']):
            np.testing.assert_allclose(a['predictions'], b['predictions'])
    assert minimal_rows[-1]['provenance']['profile_stages'] is False
    with pytest.raises(FileExistsError):
        run_evaluation(evaluation)
