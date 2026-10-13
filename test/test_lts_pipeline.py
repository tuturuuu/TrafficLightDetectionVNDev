import numpy as np
import pytest
import torch

from adaptile import Pipeline
from adaptile.detectors import YOLODetector, register_custom_modules
from adaptile.merge import NMSMerger
from adaptile.models.lts import LTSHead
from adaptile.models.lts_checkpoint import save_lts_checkpoint
from adaptile.selectors import LTSSelector, ParameterFreeSelector
from adaptile.selectors.base import AllTiles
from adaptile.tiling import UniformTiler


@pytest.fixture
def detector(tmp_path):
    from ultralytics import YOLO
    torch.set_num_threads(2)
    register_custom_modules()
    model = YOLO('configs/models/yolo26m-p2.yaml')
    path = tmp_path / 'detector.pt'
    model.save(path)
    return YOLODetector(path, imgsz=64, batch=2, device='cpu', conf=1e-6, max_det=5)


def test_keep_all_and_empty_selection_share_fine_model(detector, tmp_path):
    image = np.full((91, 135, 3), 120, np.uint8)
    tiler, merger = UniformTiler(tile_size=64, overlap=.2), NMSMerger()
    param_free = ParameterFreeSelector(feature_stage=4, coarse_imgsz=64, force_keep_all=True)
    pipe = Pipeline(tiler, param_free, detector, merger)
    assert param_free.provider.model is detector.model.model
    assert detector.model.predictor.model.backend.model is param_free.provider.model
    all_out = Pipeline(tiler, AllTiles(), detector, merger)(image)
    assert all_out.dets
    np.testing.assert_allclose(pipe(image).dets, all_out.dets)
    head_path = tmp_path / 'lts.pt'
    save_lts_checkpoint(head_path, LTSHead(param_free.provider.channels), param_free.provider,
                        detector.weights, 'frozen_head_pilot', {'purpose': 'synthetic_test'})
    lts = LTSSelector(head_path, feature_stage=4, coarse_imgsz=64, force_keep_all=True)
    lts_pipe = Pipeline(tiler, lts, detector, merger)
    np.testing.assert_allclose(lts_pipe(image).dets, all_out.dets)
    lts.force_keep_all = False
    lts.threshold = 1
    result = lts_pipe(image)
    assert result.n_kept == 0 and result.dets == []
    assert result.selection.scores.shape == (result.n_tiles,)


def test_checkpoint_pair_and_feature_mismatch_fail(detector, tmp_path):
    selector = ParameterFreeSelector(feature_stage=4, coarse_imgsz=64)
    selector.bind_detector(detector)
    path = tmp_path / 'lts.pt'
    save_lts_checkpoint(path, LTSHead(selector.provider.channels), selector.provider,
                        detector.weights, 'frozen_head_pilot', {})
    with pytest.raises(ValueError, match='feature_stage'):
        LTSSelector(path, feature_stage=2, coarse_imgsz=64).bind_detector(detector)
    changed = tmp_path / 'changed.pt'
    changed.write_bytes(b'different detector')
    detector.weights = str(changed)
    with pytest.raises(ValueError, match='SHA256'):
        LTSSelector(path, feature_stage=4, coarse_imgsz=64).bind_detector(detector)
