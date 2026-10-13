"""COCO bbox evaluation of identical original-image predictions.

This adapter does not implement dataset-specific ignore regions (e.g. VisDrone).
Keep its protocol separate from the historical, uncapped adaptile evaluator.
"""

import contextlib
import io

import numpy as np


def export_coco(predictions, targets, image_shapes):
    if not len(predictions) == len(targets) == len(image_shapes):
        raise ValueError("Predictions, targets and image shapes must align")
    categories = sorted({int(g[4]) for groups in targets for g in groups}
                        | {int(d[5]) for groups in predictions for d in groups})
    dataset = {"info": {}, "images": [], "annotations": [],
               "categories": [{"id": c, "name": str(c)} for c in categories]}
    detections = []
    for image_id, (preds, gts, shape) in enumerate(zip(predictions, targets, image_shapes)):
        height, width = shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive")
        dataset["images"].append({"id": image_id, "height": int(height), "width": int(width)})
        for rows, is_gt in ((gts, True), (preds, False)):
            for row in rows:
                if len(row) != (5 if is_gt else 6) or not np.isfinite(row).all():
                    raise ValueError("Expected finite original-image xyxy boxes with class/score")
                x1, y1, x2, y2 = map(float, row[:4])
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Boxes must have positive area")
                entry = {"image_id": image_id, "category_id": int(row[-1]),
                         "bbox": [x1, y1, x2 - x1, y2 - y1]}
                if is_gt:
                    entry.update(id=len(dataset["annotations"]) + 1, iscrowd=0,
                                 area=(x2 - x1) * (y2 - y1))
                    dataset["annotations"].append(entry)
                else:
                    entry["score"] = float(row[4])
                    detections.append(entry)
    return dataset, detections


def evaluate_coco(predictions, targets, image_shapes, max_dets=100):
    """Return official pycocotools AP; unavailable area ranges are None, not zero."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not isinstance(max_dets, int) or max_dets < 10:
        raise ValueError("COCO max_dets must be an integer >= 10")
    dataset, detections = export_coco(predictions, targets, image_shapes)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = dataset
        gt.createIndex()
        if detections:
            pred = gt.loadRes(detections)
        else:
            pred = COCO()
            pred.dataset = {**dataset, "annotations": []}
            pred.createIndex()
        evaluator = COCOeval(gt, pred, "bbox")
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDets

    def average(area="all", iou=None):
        values = precision[:, :, :, evaluator.params.areaRngLbl.index(area), -1]
        if iou is not None:
            values = values[np.isclose(evaluator.params.iouThrs, iou)]
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None

    return {"mAP50": average(iou=0.5), "mAP50-95": average(),
            "AP_S": average("small"), "AP_M": average("medium"), "AP_L": average("large"),
            "protocol": "pycocotools_bbox_original_pixels",
            "max_dets": max_dets, "area_source": "original_image_bbox"}
