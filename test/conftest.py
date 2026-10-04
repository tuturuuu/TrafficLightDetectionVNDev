"""
Fixture chung cho test.

- test_units.py  : không cần GPU/data, chạy ở mọi máy.
- test_parity.py : so pipeline mới với legacy/dense_tiling_experiment.py (code cũ,
                   làm chuẩn) trên ảnh Kayuan thật. Tự skip nếu thiếu GPU,
                   data hoặc weights.

Tùy chọn:
    pytest test/ --n-images 50     # số ảnh cho test so khớp (mặc định 50)
    pytest test/ --full            # thêm test trên toàn bộ tập test Kayuan
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "legacy"))           # code cũ làm chuẩn: dense_tiling_experiment
sys.path.insert(0, os.path.join(REPO, "legacy/evaluations"))   # evaluation_with_cnn
os.chdir(REPO)                    # đường dẫn trong configs/ là tương đối với gốc repo

KAYUAN = os.path.join(REPO, "data/kayuan/test")
DETECTOR = os.path.join(REPO, "runs/kayuan_yolov8_cbam_tiled_seed0/weights/best.pt")
GRIDNET = os.path.join(REPO, "weights/gridnet_kayuan_8x8.pth")
TILE_CNN = os.path.join(REPO, "weights/tile_proposal_cnn_kayuan.pth")


def pytest_addoption(parser):
    parser.addoption("--n-images", type=int, default=50)
    parser.addoption("--full", action="store_true",
                     help="chạy test trên toàn bộ tập test Kayuan (vài phút)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--full"):
        skip = pytest.mark.skip(reason="cần --full")
        for item in items:
            if "full" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def require_gpu_assets():
    import torch
    missing = [p for p in (KAYUAN, DETECTOR, GRIDNET, TILE_CNN) if not os.path.exists(p)]
    if missing:
        pytest.skip(f"thiếu: {missing}")
    if not torch.cuda.is_available():
        pytest.skip("không có GPU")


def _load_kayuan(n):
    import cv2
    from adaptile.data import label_path_for, list_images, load_gt

    paths = list_images(os.path.join(KAYUAN, "images"), max_images=n)
    images, gts = [], []
    for p in paths:
        img = cv2.imread(p)
        images.append(img)
        gts.append(load_gt(label_path_for(p, os.path.join(KAYUAN, "labels")),
                           *img.shape[:2]))
    return images, gts


@pytest.fixture(scope="session")
def kayuan(require_gpu_assets, request):
    return _load_kayuan(request.config.getoption("--n-images"))


@pytest.fixture(scope="session")
def kayuan_full(require_gpu_assets):
    return _load_kayuan(None)


@pytest.fixture(scope="session")
def detector(require_gpu_assets):
    from adaptile.detectors import YOLODetector
    return YOLODetector(DETECTOR, conf=0.25, imgsz=640, batch=16)
