import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("pycocotools")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from adaptile.metrics import evaluate
from adaptile.metrics_coco import evaluate_coco, export_coco


def test_identical_detections_match_official_and_existing_metric():
    gts = [[(0, 0, 10, 10, 0), (50, 50, 150, 150, 1)], [(0, 0, 40, 40, 0)]]
    preds = [[(0, 0, 10, 10, .9, 0), (51, 51, 151, 151, .8, 1),
              (100, 0, 110, 10, .5, 0)], [(0, 0, 35, 40, .7, 0)]]
    shapes = [(200, 300), (200, 300)]
    dataset, detections = export_coco(preds, gts, shapes)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = dataset; gt.createIndex()
        reference = COCOeval(gt, gt.loadRes(detections), 'bbox')
        reference.evaluate(); reference.accumulate(); reference.summarize()
    result = evaluate_coco(preds, gts, shapes)
    assert result['mAP50-95'] == pytest.approx(reference.stats[0])
    assert result['mAP50'] == pytest.approx(reference.stats[1])
    assert result['AP_S'] == pytest.approx(reference.stats[3])
    historical = evaluate(preds, gts)
    assert result['mAP50'] == pytest.approx(historical['mAP50'])
    assert result['mAP50-95'] == pytest.approx(historical['mAP50-95'])


def test_small_area_uses_original_pixels_not_tile_resize():
    gts = [[(0, 0, 10, 10, 0), (50, 50, 150, 150, 0)]]
    result = evaluate_coco([[(50, 50, 150, 150, .9, 0)]], gts, [(1080, 1920)])
    assert result['AP_S'] == 0
    assert result['AP_L'] == pytest.approx(1)
    assert result['AP_M'] is None


def test_empty_predictions_and_input_validation():
    assert evaluate_coco([[]], [[(0, 0, 10, 10, 0)]], [(20, 20)])['mAP50'] == 0
    with pytest.raises(ValueError, match='align'):
        export_coco([], [[]], [])
    with pytest.raises(ValueError, match='finite'):
        export_coco([[(0, 0, 1, 1, np.nan, 0)]], [[]], [(20, 20)])


def test_historical_uncapped_protocol_remains_distinct():
    preds = [[(100, 100, 110, 110, 1 - i / 1000, 0) for i in range(100)]
             + [(0, 0, 10, 10, .1, 0)]]
    gts = [[(0, 0, 10, 10, 0)]]
    assert evaluate_coco(preds, gts, [(200, 200)])['mAP50'] == 0
    assert evaluate(preds, gts)['mAP50'] > 0
