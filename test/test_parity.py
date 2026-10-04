"""
Test so khớp với code cũ trên ảnh Kayuan thật (mục 4, bước 2 của HANDOFF.md).

Chuẩn là legacy/dense_tiling_experiment.py: pipeline mới phải cho prediction GIỐNG
HỆT (cùng box, score, class, thứ tự) trên --n-images ảnh đầu của tập test.
Metric chỉ được khác ở phần xét class.

    CUDA_VISIBLE_DEVICES=1 pytest test/test_parity.py -v
    CUDA_VISIBLE_DEVICES=1 pytest test/test_parity.py -v --full   # + số ở mục 3.1
"""

import pytest

import dense_tiling_experiment as legacy
from adaptile.merge import NMSMerger
from adaptile.metrics import evaluate
from adaptile.pipeline import Pipeline, run_dataset
from adaptile.selectors import AllTiles, GridNetSelector
from adaptile.tiling import UniformTiler

from conftest import GRIDNET

THRESHOLD = 0.10          # threshold GridNet đã dùng cho bảng ở mục 3.1 / 3.3
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def selectors():
    return {"uniform": AllTiles(),
            "gridnet": GridNetSelector(GRIDNET, threshold=THRESHOLD)}


@pytest.fixture(scope="module")
def legacy_gridnet():
    # GridProposalNet chỉ còn một bản (adaptile.models.gridnet, chép nguyên văn từ
    # grid_proposal_net.py cũ); load riêng một instance để vòng lặp cũ dùng.
    from adaptile.models.gridnet import GridProposalNet
    return GridProposalNet.load(GRIDNET)


def legacy_predict(img, tile_size, model, gridnet=None):
    """Nguyên văn vòng lặp trong dense_tiling_experiment.run(), bỏ phần đo giờ."""
    H, W = img.shape[:2]
    tiles, _, _ = legacy.make_grid(H, W, tile_size, 0.2)
    if gridnet is not None:
        keep_coarse, _ = gridnet.propose(img, threshold=THRESHOLD)
        keep = legacy.grid_mask_for_tiles(keep_coarse, tiles, H, W,
                                          gridnet.grid_rows, gridnet.grid_cols)
        tiles_run = [t for t, k in zip(tiles, keep) if k] or tiles[:1]
    else:
        tiles_run = tiles
    dets = legacy.yolo_on_tiles(model, img, tiles_run, 0.25, 640)
    return legacy.nms_merge(dets, 0.5), len(tiles_run)


def make_pipe(tile_size, selector, detector):
    return Pipeline(UniformTiler(tile_size, 0.2), selector, detector, NMSMerger(0.5))


@pytest.mark.parametrize("tile_size", [640, 320])
@pytest.mark.parametrize("method", ["uniform", "gridnet"])
def test_predictions_identical(kayuan, detector, selectors, legacy_gridnet,
                               tile_size, method):
    images, gts = kayuan
    pipe = make_pipe(tile_size, selectors[method], detector)
    lg = legacy_gridnet if method == "gridnet" else None

    new_preds, old_preds = [], []
    for i, img in enumerate(images):
        out = pipe(img)
        old, n_kept = legacy_predict(img, tile_size, detector.model, lg)
        assert out.n_kept == n_kept, f"ảnh {i}: số tile giữ lại khác nhau"
        assert out.dets == old, f"ảnh {i}: prediction khác nhau"
        new_preds.append(out.dets)
        old_preds.append(old)
    assert sum(map(len, new_preds)) > 0, "không có detection nào, test vô nghĩa"

    # metric: chế độ không xét class phải ra đúng số của evaluate_accuracy cũ
    old_gts = [[g[:4] for g in G] for G in gts]
    ap50, p, r = legacy.evaluate_accuracy(old_preds, old_gts)
    m = evaluate(new_preds, gts, class_aware=False)
    assert (m["mAP50"], m["precision"], m["recall"]) == (ap50, p, r)


def test_run_dataset_matches_per_image(kayuan, detector, selectors):
    """run_dataset (có warmup + đo giờ) không làm thay đổi prediction."""
    images, _ = kayuan
    pipe = make_pipe(640, selectors["gridnet"], detector)
    run = run_dataset(pipe, images[:10])
    assert run.preds == [pipe(img).dets for img in images[:10]]
    assert run.total_tiles == 8 * 10 and 0 < run.kept_tiles <= run.total_tiles


@pytest.mark.full
@pytest.mark.parametrize("method,expected", [
    # (mAP50, R) cách cũ không xét class  |  (mAP50, R) có xét class — HANDOFF.md mục 3.1
    ("uniform", ((0.9637, 0.9942), (0.9697, 0.9874))),
    ("gridnet", ((0.9637, 0.9935), (0.9692, 0.9868))),
])
def test_full_kayuan_matches_handoff_numbers(kayuan_full, detector, selectors,
                                             method, expected):
    images, gts = kayuan_full
    run = run_dataset(make_pipe(640, selectors[method], detector), images)
    old = evaluate(run.preds, gts, class_aware=False)
    new = evaluate(run.preds, gts, class_aware=True)
    assert (round(old["mAP50"], 4), round(old["recall"], 4)) == expected[0]
    assert (round(new["mAP50"], 4), round(new["recall"], 4)) == expected[1]
    print(f"\n{method}: class-aware mAP50={new['mAP50']:.4f} "
          f"mAP50-95={new['mAP50-95']:.4f} P={new['precision']:.4f} "
          f"R={new['recall']:.4f}, kept/img={run.kept_tiles / len(images):.2f}")


# ── tile_cnn: so với legacy/evaluations/evaluation_with_cnn.py ─────────────────────

@pytest.mark.parametrize("tile_size", [640, 320])
def test_tile_cnn_selection_identical(kayuan, tile_size):
    """Điểm CNN và tile được chọn phải giống code cũ. (Phần YOLO/NMS của code cũ
    khác pipeline chuẩn ở batch size, max_det, nên chỉ so khâu chọn tile.)"""
    import numpy as np
    import torch
    import evaluation_with_cnn as old_cnn
    from adaptile.selectors import TileCNNSelector
    from conftest import TILE_CNN

    images, _ = kayuan
    new = TileCNNSelector(TILE_CNN, threshold=0.5)
    old_model = old_cnn.load_tile_model(TILE_CNN, torch.device("cuda:0"))
    n_kept = 0
    for i, img in enumerate(images):
        tiles = UniformTiler(tile_size, 0.2).tiles(*img.shape[:2])
        old_tiles = old_cnn.build_tiles(img, tile_size, 0.2)
        assert [(t[0], t[1]) for t in tiles] == [(t["x"], t["y"]) for t in old_tiles]
        old_scores = old_cnn.tile_model_scores_batched(
            old_model, [t["image"] for t in old_tiles], torch.device("cuda:0"),
            min(len(old_tiles), 64))
        new_scores = new.scores(img, tiles)
        np.testing.assert_array_equal(new_scores, old_scores, err_msg=f"ảnh {i}")
        np.testing.assert_array_equal(new.select(img, tiles), old_scores > 0.5)
        n_kept += int((old_scores > 0.5).sum())
    assert n_kept > 0
