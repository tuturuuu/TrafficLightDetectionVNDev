"""
Test phần train (adaptile.training, adaptile.dataset_tiling).

Phần lớn chạy trên dataset giả nhỏ, không cần GPU. Test train thật 1 epoch
(detector + GridNet) chỉ chạy với --full.
"""

import json

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from adaptile.dataset_tiling import ensure_tiled, tile_image_and_labels
from adaptile.training import _label_dir, prepare_detector_data, resolve_arch

CLASS_NAMES = {0: "a", 1: "b"}


def fake_dataset(root, n=4, size=(720, 1280), seed=0):
    """Dataset YOLO giả kiểu Kayuan: <split>/images, <split>/labels, data.yaml."""
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)
            cv2.imwrite(str(root / split / "images" / f"im{i}.jpg"), img)
            lines = [f"{rng.integers(2)} {rng.uniform(0.1, 0.9):.4f} {rng.uniform(0.1, 0.9):.4f} 0.02 0.04"
                     for _ in range(5)]
            (root / split / "labels" / f"im{i}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text(
        "train: train/images\nval: val/images\nnames:\n  0: a\n  1: b\n")
    return root / "data.yaml"


def det_cfg(data_yaml, **tiling):
    return OmegaConf.create({
        "dataset": {"data_yaml": str(data_yaml)},
        "tiling": {"type": "uniform", "tile_size": 640, "overlap": 0.2,
                   "min_visibility": 0.3, "suffix": "_tiled_uniform", **tiling},
    })


# ── kiến trúc ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("arch,att,expected", [
    ("yolov8m", "cbam", "configs/models/YOLOv8_CBAM.yaml"),
    ("yolov8m", "none", "yolov8m.yaml"),
    ("yolo26m", "se", "configs/models/yolo26m-modified.yaml"),
    ("my/arch.yaml", "cbam", "my/arch.yaml"),
])
def test_resolve_arch(arch, att, expected):
    assert resolve_arch(OmegaConf.create({"arch": arch, "attention": att})) == expected


def test_resolve_arch_unknown_lists_options():
    with pytest.raises(KeyError, match="yolov8m\\+cbam"):
        resolve_arch(OmegaConf.create({"arch": "yolov8m", "attention": "se"}))


def test_label_dir_both_layouts():
    assert str(_label_dir("data/kayuan/train/images")) == "data/kayuan/train/labels"
    assert str(_label_dir("data/SDS/images/train")) == "data/SDS/labels/train"


# ── cắt tile ────────────────────────────────────────────────────────────────

def test_tile_labels_are_clipped_into_tile(tmp_path):
    img = np.zeros((720, 1280, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), img)
    # box nằm vắt qua ranh giới hai tile theo chiều ngang
    (tmp_path / "a.txt").write_text("1 0.5 0.5 0.05 0.05\n")
    (tmp_path / "oi").mkdir(); (tmp_path / "ol").mkdir()
    tile_image_and_labels(str(tmp_path / "a.jpg"), str(tmp_path / "a.txt"),
                          str(tmp_path / "oi"), str(tmp_path / "ol"), 640, 0.2, 0.3)
    tiles = sorted((tmp_path / "oi").iterdir())
    assert len(tiles) == 3 * 2                                   # x: 0,512,640; y: 0,80
    assert all(cv2.imread(str(t)).shape == (640, 640, 3) for t in tiles)
    for f in (tmp_path / "ol").iterdir():
        for line in f.read_text().split("\n")[:-1]:
            c, *xywh = line.split()
            assert c == "1" and all(0 <= float(v) <= 1 for v in xywh)


def test_ensure_tiled_creates_reuses_and_rejects_mismatch(tmp_path):
    data_yaml = fake_dataset(tmp_path)
    img_dir, lbl_dir = tmp_path / "train/images", tmp_path / "train/labels"
    out = tmp_path / "train_tiled_uniform"
    assert ensure_tiled(img_dir, lbl_dir, out, 640, 0.2, 0.3) == "created"
    assert json.loads((out / "tiling.json").read_text())["tile_size"] == 640
    assert len(list((out / "images").iterdir())) == 4 * 6
    assert ensure_tiled(img_dir, lbl_dir, out, 640, 0.2, 0.3) == "reused"
    with pytest.raises(RuntimeError, match="tham số khác"):
        ensure_tiled(img_dir, lbl_dir, out, 320, 0.2, 0.3)


def test_ensure_tiled_reuses_old_dir_without_marker(tmp_path):
    out = tmp_path / "old_tiled"
    (out / "images").mkdir(parents=True)
    (out / "images" / "x.jpg").write_bytes(b"")
    assert ensure_tiled(tmp_path, tmp_path, out, 640, 0.2, 0.3) == "reused-unverified"


def test_prepare_detector_data(tmp_path):
    data_yaml = fake_dataset(tmp_path)
    assert prepare_detector_data(det_cfg(data_yaml, type="none")) == str(data_yaml)
    tiled = prepare_detector_data(det_cfg(data_yaml))
    y = OmegaConf.load(tiled)
    assert tiled.endswith("data_tiled_uniform.yaml")
    assert y.train == "train_tiled_uniform/images" and y.val == "val_tiled_uniform/images"
    assert dict(y.names) == CLASS_NAMES
    assert (tmp_path / "val_tiled_uniform/labels").is_dir()


def test_all_train_configs_parse():
    from adaptile.training import train_detector, train_selector
    for name in ("detector_kayuan", "detector_seadronesee"):
        cfg = OmegaConf.load(f"configs/train/{name}.yaml")
        resolve_arch(cfg.model)
        train_detector(cfg, dry_run=True)
    for name in ("selector_kayuan", "selector_seadronesee"):
        for kind in ("gridnet", "tile_cnn"):
            cfg = OmegaConf.load(f"configs/train/{name}.yaml")
            cfg.type = kind
            train_selector(cfg, dry_run=True)


# ── train thật 1 epoch (chậm) ───────────────────────────────────────────────

@pytest.mark.full
@pytest.mark.gpu
def test_train_detector_one_epoch(tmp_path):
    from adaptile.training import train_detector
    data_yaml = fake_dataset(tmp_path / "ds", n=4)
    cfg = OmegaConf.merge(det_cfg(data_yaml), {
        "model": {"arch": "yolov8m", "attention": "cbam", "pretrained": None},
        "train": {"seeds": [0], "epochs": 1, "imgsz": 320, "batch": 4, "patience": 5,
                  "workers": 0, "device": 0, "deterministic": True,
                  "extra": {"plots": False, "amp": False}},
        "output": {"project": str(tmp_path / "runs"), "name": "smoke"},
    })
    [run] = train_detector(cfg)
    assert (tmp_path / "runs/smoke_seed0/weights/best.pt").exists(), run


@pytest.mark.full
@pytest.mark.gpu
def test_train_gridnet_one_epoch(tmp_path):
    from adaptile.models.gridnet import GridProposalNet, train_gridnet
    fake_dataset(tmp_path / "ds", n=6)
    out = tmp_path / "g.pth"
    train_gridnet(str(tmp_path / "ds/train/images"), str(tmp_path / "ds/train/labels"),
                  str(out), epochs=1, batch_size=2, num_workers=0)
    net = GridProposalNet.load(str(out))
    keep, probs = net.propose(np.zeros((720, 1280, 3), np.uint8))
    assert keep.shape == probs.shape == (8, 8)
