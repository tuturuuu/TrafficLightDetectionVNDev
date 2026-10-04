"""
Test đơn vị: mỗi hàm lõi của adaptile phải cho kết quả giống hệt bản cũ
trong legacy/dense_tiling_experiment.py trên input ngẫu nhiên. Không cần GPU/data.
"""

import numpy as np
import pytest

import dense_tiling_experiment as legacy
from adaptile import metrics, registry
from adaptile.merge import iou, nms_merge
from adaptile.tiling import NoTiler, UniformTiler, grid_mask_for_tiles, make_grid

RNG = np.random.default_rng(0)
SIZES = [(1080, 1920), (2160, 3840), (933, 1230), (3632, 5456), (500, 700)]


def random_dets(n, H=1080, W=1920, n_cls=4, rng=RNG):
    out = []
    for _ in range(n):
        x1, y1 = rng.uniform(0, W - 60), rng.uniform(0, H - 60)
        w, h = rng.uniform(5, 60, size=2)
        out.append((np.float32(x1), np.float32(y1), np.float32(x1 + w),
                    np.float32(y1 + h), float(rng.uniform(0.25, 1)),
                    int(rng.integers(n_cls))))
    return out


def jitter_gts(dets, rng=RNG, p_drop=0.2, p_flip=0.1, n_cls=4):
    """GT sinh từ prediction: dịch nhẹ, bỏ bớt, đổi class -> có cả TP lẫn FP/FN."""
    gts = []
    for d in dets:
        if rng.random() < p_drop:
            continue
        dx, dy = rng.normal(0, 3, size=2)
        cls = int(rng.integers(n_cls)) if rng.random() < p_flip else d[5]
        gts.append((d[0] + dx, d[1] + dy, d[2] + dx, d[3] + dy, cls))
    return gts + [(*random_dets(1, rng=rng)[0][:4], int(rng.integers(n_cls)))
                  for _ in range(rng.integers(0, 3))]


# ── tiling ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("H,W", SIZES)
@pytest.mark.parametrize("tile_size", [160, 240, 320, 480, 640, 1280])
@pytest.mark.parametrize("overlap", [0.0, 0.2, 0.5])
def test_make_grid_same_as_legacy(H, W, tile_size, overlap):
    assert make_grid(H, W, tile_size, overlap) == legacy.make_grid(H, W, tile_size, overlap)


@pytest.mark.parametrize("H,W", SIZES)
@pytest.mark.parametrize("tile_size", [160, 320, 640])
def test_grid_mask_same_as_legacy(H, W, tile_size):
    tiles, _, _ = make_grid(H, W, tile_size)
    for _ in range(20):
        keep = RNG.random((8, 8)) < RNG.uniform(0.05, 0.6)
        np.testing.assert_array_equal(
            grid_mask_for_tiles(keep, tiles, H, W, 8, 8),
            legacy.grid_mask_for_tiles(keep, tiles, H, W, 8, 8))


def test_uniform_tiler_crop_and_pad():
    img = RNG.integers(0, 255, size=(500, 700, 3), dtype=np.uint8)
    tiles = UniformTiler(tile_size=640).tiles(500, 700)
    assert [c.shape for c in UniformTiler(640).crop(img, tiles)] == [(500, 640, 3)] * 2
    padded = UniformTiler(640, pad=True).crop(img, tiles)
    assert [c.shape for c in padded] == [(640, 640, 3)] * 2
    assert (padded[0][500:] == 114).all()


def test_no_tiler_is_whole_image():
    assert NoTiler().tiles(1080, 1920) == [(0, 0, 1920, 1080, 0, 0)]


# ── merge ───────────────────────────────────────────────────────────────────

def test_iou_same_as_legacy():
    a, b = random_dets(200), random_dets(200)
    for x, y in zip(a, b):
        assert iou(x, y) == legacy._iou(x, y)


@pytest.mark.parametrize("seed", range(5))
def test_nms_same_as_legacy(seed):
    rng = np.random.default_rng(seed)
    base = random_dets(80, rng=rng)
    # thêm bản sao lệch nhẹ để NMS thực sự phải loại box (như tile chồng nhau)
    dets = base + [(d[0] + 2, d[1] + 1, d[2] + 2, d[3] + 1, d[4] * 0.9, d[5])
                   for d in base[:40]]
    assert nms_merge(dets, 0.5) == legacy.nms_merge(dets, 0.5)


def test_nms_max_det_keeps_top_scores():
    dets = random_dets(50)
    out = nms_merge(dets, 0.5, max_det=10)
    assert len(out) == 10
    assert min(d[4] for d in out) >= sorted(d[4] for d in nms_merge(dets, 0.5))[-10]


# ── metrics ─────────────────────────────────────────────────────────────────

def eval_class_aware_reference(preds, gts, iou_thr=0.5):
    """Hàm kiểm chứng ở Phụ lục A của HANDOFF.md (nguyên văn)."""
    def ap101(records, n_gt):
        if not records or n_gt == 0:
            return 0.0, 0.0, 0.0
        records.sort(key=lambda r: r[0], reverse=True)
        tps = np.cumsum([r[1] for r in records]); fps = np.cumsum([not r[1] for r in records])
        prec = tps / (tps + fps); rec = tps / n_gt
        ap = np.mean([(prec[rec >= t].max() if (rec >= t).any() else 0.0)
                      for t in np.linspace(0, 1, 101)])
        return float(ap), float(prec[-1]), float(rec[-1])

    classes = sorted({g[4] for gg in gts for g in gg})
    aps, tp_all, n_pred, n_gt_all = [], 0, 0, 0
    for c in classes:
        recs, n_gt = [], 0
        for P, G in zip(preds, gts):
            g = [x for x in G if x[4] == c]; n_gt += len(g); m = [False] * len(g)
            for d in sorted([d for d in P if d[5] == c], key=lambda d: -d[4]):
                best, bi = 0.0, -1
                for j, gb in enumerate(g):
                    if not m[j] and (i := legacy._iou(d, gb)) > best:
                        best, bi = i, j
                ok = best >= iou_thr and bi >= 0
                if ok: m[bi] = True
                recs.append((d[4], ok))
        ap, _, _ = ap101(recs, n_gt); aps.append(ap)
        tp_all += sum(r[1] for r in recs); n_pred += len(recs); n_gt_all += n_gt
    n_pred += sum(1 for P in preds for d in P if d[5] not in classes)
    return float(np.mean(aps)), tp_all / max(n_pred, 1), tp_all / max(n_gt_all, 1)


def synthetic_dataset(seed, n_img=30):
    rng = np.random.default_rng(seed)
    preds = [random_dets(int(rng.integers(0, 15)), rng=rng) for _ in range(n_img)]
    gts = [jitter_gts(P, rng=rng) for P in preds]
    return preds, gts


@pytest.mark.parametrize("seed", range(10))
def test_eval_class_agnostic_same_as_legacy(seed):
    preds, gts = synthetic_dataset(seed)
    new = metrics.evaluate(preds, gts, class_aware=False)
    ap50, p, r = legacy.evaluate_accuracy(preds, [[g[:4] for g in G] for G in gts])
    assert (new["mAP50"], new["precision"], new["recall"]) == (ap50, p, r)


@pytest.mark.parametrize("seed", range(10))
def test_eval_class_aware_same_as_appendix(seed):
    preds, gts = synthetic_dataset(seed)
    new = metrics.evaluate(preds, gts, class_aware=True)
    ap50, p, r = eval_class_aware_reference(preds, gts)
    assert new["mAP50"] == pytest.approx(ap50, abs=1e-12)
    assert (new["precision"], new["recall"]) == (p, r)


def test_eval_perfect_predictions():
    gts = [[(10, 10, 50, 50, 0), (100, 100, 140, 160, 1)], [(5, 5, 20, 20, 1)]]
    preds = [[(*g[:4], 0.9, g[4]) for g in G] for G in gts]
    m = metrics.evaluate(preds, gts)
    assert m["mAP50"] == m["mAP50-95"] == m["precision"] == m["recall"] == 1.0


def test_eval_wrong_class_is_fp_only_when_class_aware():
    gts = [[(10, 10, 50, 50, 0)]]
    preds = [[(10, 10, 50, 50, 0.9, 3)]]
    assert metrics.evaluate(preds, gts, class_aware=False)["recall"] == 1.0
    m = metrics.evaluate(preds, gts, class_aware=True)
    assert (m["recall"], m["precision"], m["mAP50"]) == (0.0, 0.0, 0.0)


def test_eval_empty():
    m = metrics.evaluate([[], []], [[], []])
    assert (m["mAP50"], m["precision"], m["recall"]) == (0.0, 0.0, 0.0)


# ── registry ────────────────────────────────────────────────────────────────

def test_registry_build_from_omegaconf():
    from omegaconf import OmegaConf
    t = registry.TILERS.build(OmegaConf.create({"name": "uniform", "tile_size": 320}))
    assert isinstance(t, UniformTiler) and t.tile_size == 320
    with pytest.raises(KeyError, match="Có sẵn"):
        registry.TILERS.build({"name": "khong_co"})
