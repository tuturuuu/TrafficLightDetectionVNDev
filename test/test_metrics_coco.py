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


@pytest.mark.parametrize('degenerate', [(0, 0, 0, 10), (0, 0, 10, 0), (10, 10, 10, 10)])
def test_zero_area_prediction_remains_false_positive_like_official_coco(degenerate):
    gts = [[(0, 0, 10, 10, 0)]]
    preds = [[(*degenerate, .99, 0), (0, 0, 10, 10, .9, 0)]]
    dataset, detections = export_coco(preds, gts, [(20, 20)])
    assert len(detections) == 2
    assert detections[0]['bbox'][2] * detections[0]['bbox'][3] == 0
    # Construct the official reference independently of the adapter's export.
    reference_dataset = {
        'info': {}, 'images': [{'id': 0, 'height': 20, 'width': 20}],
        'categories': [{'id': 0, 'name': '0'}],
        'annotations': [{'id': 1, 'image_id': 0, 'category_id': 0,
                         'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0}],
    }
    x1, y1, x2, y2 = degenerate
    reference_detections = [
        {'image_id': 0, 'category_id': 0, 'bbox': [x1, y1, x2-x1, y2-y1], 'score': .99},
        {'image_id': 0, 'category_id': 0, 'bbox': [0, 0, 10, 10], 'score': .9},
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = reference_dataset
        gt.createIndex()
        reference = COCOeval(gt, gt.loadRes(reference_detections), 'bbox')
        reference.evaluate()
        reference.accumulate()
        reference.summarize()
    result = evaluate_coco(preds, gts, [(20, 20)])
    assert result['mAP50'] == pytest.approx(.5)
    assert result['mAP50-95'] == pytest.approx(.5)
    assert result['AP_S'] == pytest.approx(.5)
    assert result['mAP50-95'] == pytest.approx(reference.stats[0])
    assert result['mAP50'] == pytest.approx(reference.stats[1])
    assert result['AP_S'] == pytest.approx(reference.stats[3])
    operating = evaluate(preds, gts)
    assert operating['precision'] == pytest.approx(.5)
    assert operating['recall'] == 1


@pytest.mark.parametrize('box', [(0, 0, 0, 10), (0, 0, 10, 0), (10, 10, 10, 10)])
def test_zero_area_ground_truth_is_rejected(box):
    with pytest.raises(ValueError, match='Ground-truth boxes must have positive area'):
        export_coco([[]], [[(*box, 0)]], [(20, 20)])


@pytest.mark.parametrize('box', [(10, 0, 0, 10), (0, 10, 10, 0), (10, 10, 0, 0)])
@pytest.mark.parametrize('is_gt', [False, True])
def test_inverted_boxes_are_rejected(box, is_gt):
    preds, gts = ([[]], [[(*box, 0)]]) if is_gt else ([[(*box, .9, 0)]], [[]])
    with pytest.raises(ValueError, match='negative extents'):
        export_coco(preds, gts, [(20, 20)])
