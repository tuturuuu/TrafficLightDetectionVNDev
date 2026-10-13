import numpy as np
import pytest
import torch

from adaptile.models.coarse_features import CoarseGeometry, letterbox
from adaptile.relevance_targets import relevance_targets
from adaptile.selectors.relevance_utils import score_tiles, select_scores


def test_overlap_targets_not_centers_and_boundary_convention():
    geometry = CoarseGeometry((64, 64), (64, 64), (64, 64), (0, 0))
    target, valid = relevance_targets([[(15, 15, 17, 17), (32, 32, 48, 48)]], [geometry], (4, 4))
    assert target.sum() == 5
    assert target[0, 0, :2, :2].sum() == 4
    assert target[0, 0, 2, 2] == 1
    assert target[0, 0, 3, 3] == 0
    assert valid.sum() == 16


@pytest.mark.parametrize('shape', [(100, 200), (200, 100), (17, 113), (113, 17)])
def test_letterbox_padding_and_all_image_coverage(shape):
    h, w = shape
    _, geometry = letterbox(np.zeros((h, w, 3), np.uint8), 64)
    target, valid = relevance_targets([[(0, 0, w, h)]], [geometry], (8, 8))
    torch.testing.assert_close(target, valid)
    assert 0 < valid.sum() < 64
    empty, _ = relevance_targets([[]], [geometry], (8, 8))
    assert empty.sum() == 0


def test_max_scores_order_strict_threshold_and_zero_selection():
    geometry = CoarseGeometry((64, 64), (64, 64), (64, 64), (0, 0))
    heat = np.zeros((4, 4), np.float32); heat[1, 1] = .8; heat[0, 0] = .2
    tiles = [(16, 16, 32, 32, 0, 0), (0, 0, 16, 16, 0, 1), (48, 48, 64, 64, 0, 2)]
    scores, fallback = score_tiles(heat, tiles, geometry)
    np.testing.assert_allclose(scores, [.8, .2, 0])
    assert fallback == 0
    assert select_scores(scores, float(scores[0])).tolist() == [False, False, False]
    assert select_scores(scores, 0).tolist() == [True, True, False]
    assert select_scores(scores, 1, min_keep=1).tolist() == [True, False, False]
    assert select_scores(scores, 1, force_keep_all=True).all()


def test_subpixel_degenerate_and_empty_tiles():
    geometry = CoarseGeometry((64, 64), (64, 64), (64, 64), (0, 0))
    heat = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    scores, fallback = score_tiles(heat, [(16.01, 16.01, 16.02, 16.02), (64, 64, 64, 64)], geometry)
    np.testing.assert_allclose(scores, [5 / 16, 15 / 16])
    assert fallback == 1
    assert score_tiles(heat, [], geometry)[0].size == 0
