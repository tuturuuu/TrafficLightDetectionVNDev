"""Regression checks for grouped splits, P2 head routing and measured stages."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from adaptile import Pipeline, run_dataset
from adaptile.detectors import register_custom_modules
from adaptile.selectors.base import AllTiles

spec = importlib.util.spec_from_file_location('split_audit', Path('scripts/data/check_split_leakage.py'))
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_grouped_split_keeps_whole_videos_and_all_frames():
    samples = [('train', str(g), Path(f'{g}_{i}.jpg'), Path(f'{g}_{i}.txt'))
               for g in range(15) for i in range(g + 1)]
    result = audit_module.grouped_split(samples)
    flattened = [x for members in result.values() for x in members]
    assert len(flattened) == len(samples)
    assert {x[2] for x in flattened} == {x[2] for x in samples}
    assert not audit_module.audit(flattened)['leakage']
    assert result == audit_module.grouped_split(list(reversed(samples)))


def test_kayuan_video_key():
    assert audit_module.sequence_key('2019012208510010734') == '20190122085100'
    assert audit_module.sequence_key('20200119_091933212') == '20200119091933'
    with pytest.raises(ValueError):
        audit_module.sequence_key('unknown')


@pytest.mark.parametrize('attention,refs', [('p2', [19, 22, 25, 28]),
                                           ('se', [22, 25, 28, 31]),
                                           ('cbam', [22, 25, 28, 31])])
def test_yolo26_four_scales_forward(attention, refs):
    from ultralytics import YOLO
    register_custom_modules()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = YOLO(f'configs/models/yolo26m-{attention}.yaml').model.eval()
        assert model.model[-1].f == refs
        assert model.stride.tolist() == [4, 8, 16, 32]
        with torch.no_grad():
            model(torch.zeros(1, 3, 64, 64))
    finally:
        torch.set_num_threads(old_threads)


def test_dataset_timings_measure_each_image(monkeypatch):
    import adaptile.pipeline as pipeline
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    class Tiler:
        def tiles(self, h, w):
            return [(0, 0, w, h, 0, 0)]
        def crop(self, image, tiles):
            return [image] * len(tiles)
    pipe = Pipeline(Tiler(), AllTiles(), lambda crops, offsets: [], lambda boxes: boxes)
    run = run_dataset(pipe, [np.zeros((8, 8, 3), dtype=np.uint8)] * 3, n_warmup=1)
    assert all(len(values) == 3 for values in run.latency_s.values())
    assert run.merge_s == sum(run.latency_s['merge'])
    assert run.detect_s == sum(run.latency_s['detect'])
    assert run.peak_gpu_memory_mb == {}
    for i in range(3):
        assert run.latency_s['total'][i] >= sum(run.latency_s[k][i] for k in ('select', 'detect', 'merge'))
