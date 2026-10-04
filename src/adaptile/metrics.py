"""
Hàm eval duy nhất của pipeline: mAP50, mAP50-95, precision, recall.

- Prediction: (x1, y1, x2, y2, score, cls); GT: (x1, y1, x2, y2, cls).
- Ghép greedy trong từng ảnh: prediction score cao trước, lấy GT chưa ghép có
  IoU lớn nhất; đúng nếu IoU >= ngưỡng. AP nội suy 101 điểm, giống
  dense_tiling_experiment.evaluate_accuracy.
- class_aware=True (mặc định): chỉ ghép cùng class, AP tính riêng từng class
  có trong GT rồi lấy trung bình. Prediction thuộc class không có trong GT
  được tính là FP khi tính precision.
- class_aware=False: gộp mọi class làm một. Cho kết quả giống hệt
  evaluate_accuracy cũ (mục 3.1 HANDOFF.md); chỉ dùng để so sánh với số cũ.
"""

import numpy as np

from .merge import iou

IOU_THRS = tuple(np.linspace(0.5, 0.95, 10))


def ap101(records, n_gt):
    """records: list (score, is_tp). Trả (AP, precision, recall) ở cuối đường cong."""
    if not records or n_gt == 0:
        return 0.0, 0.0, 0.0
    records = sorted(records, key=lambda r: r[0], reverse=True)
    tps = np.cumsum([r[1] for r in records])
    fps = np.cumsum([not r[1] for r in records])
    prec = tps / (tps + fps)
    rec = tps / n_gt
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t]
        ap += (p.max() if len(p) else 0.0)
    ap /= 101
    return float(ap), float(prec[-1]), float(rec[-1])


def _match_image(preds, gts, iou_thr):
    """Greedy matching trong một ảnh, một class. Trả list (score, is_tp)."""
    matched = [False] * len(gts)
    out = []
    for d in sorted(preds, key=lambda d: d[4], reverse=True):
        best, bi = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            i = iou(d, g)
            if i > best:
                best, bi = i, j
        ok = best >= iou_thr and bi >= 0
        if ok:
            matched[bi] = True
        out.append((d[4], ok))
    return out


def evaluate(all_preds, all_gts, class_aware=True, iou_thrs=IOU_THRS):
    """all_preds / all_gts: list theo ảnh. Precision/recall lấy ở iou_thrs[0]."""
    if not class_aware:
        all_preds = [[(*d[:5], 0) for d in P] for P in all_preds]
        all_gts = [[(*g[:4], 0) for g in G] for G in all_gts]

    classes = sorted({g[4] for G in all_gts for g in G})
    ap = np.zeros((len(classes), len(iou_thrs)))
    tp, n_pred, n_gt_all = 0, 0, 0
    for ci, c in enumerate(classes):
        P_c = [[d for d in P if d[5] == c] for P in all_preds]
        G_c = [[g for g in G if g[4] == c] for G in all_gts]
        n_gt = sum(len(g) for g in G_c)
        for ti, thr in enumerate(iou_thrs):
            records = []
            for P, G in zip(P_c, G_c):
                records += _match_image(P, G, thr)
            ap[ci, ti] = ap101(records, n_gt)[0]
            if ti == 0:
                tp += sum(r[1] for r in records)
                n_pred += len(records)
        n_gt_all += n_gt
    # prediction của class không có trong GT: toàn bộ là FP
    n_pred += sum(1 for P in all_preds for d in P if d[5] not in classes)

    return {
        "mAP50": float(ap[:, 0].mean()) if classes else 0.0,
        "mAP50-95": float(ap.mean()) if classes else 0.0,
        "precision": tp / n_pred if n_pred else 0.0,
        "recall": tp / n_gt_all if n_gt_all else 0.0,
        "ap50_per_class": {int(c): float(ap[i, 0]) for i, c in enumerate(classes)},
    }
