"""Gộp detection từ các tile: NMS theo class.

Detection là tuple (x1, y1, x2, y2, score, cls) theo pixel ảnh gốc.
`iou` và `nms_merge` chuyển nguyên văn từ legacy/dense_tiling_experiment.py.
"""

from .registry import MERGERS


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


def nms_merge(dets, iou_thr=0.5, max_det=None):
    """NMS theo từng class. `max_det` (vd 300 như evaluation_with_cnn.py) giữ
    lại max_det box có score cao nhất; None = không giới hạn (mặc định cũ)."""
    if not dets:
        return []
    out = []
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d[5], []).append(d)
    for cls, ds in by_cls.items():
        ds = sorted(ds, key=lambda d: d[4], reverse=True)
        while ds:
            best = ds.pop(0)
            out.append(best)
            ds = [d for d in ds if iou(best, d) < iou_thr]
    if max_det and len(out) > max_det:
        out = sorted(out, key=lambda d: d[4], reverse=True)[:max_det]
    return out


@MERGERS.register("nms")
class NMSMerger:
    def __init__(self, iou_thr=0.5, max_det=None):
        self.iou_thr = iou_thr
        self.max_det = max_det

    def __call__(self, dets):
        return nms_merge(dets, self.iou_thr, self.max_det)


@MERGERS.register("none")
class NoMerge:
    """Không gộp: giữ mọi detection của mọi tile (box trùng ở vùng overlap còn nguyên)."""

    def __call__(self, dets):
        return list(dets)
