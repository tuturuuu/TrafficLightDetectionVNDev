"""Test đọc file config (configs/*.yaml + adaptile.config). Không cần GPU/data."""

import pytest

from adaptile import MERGERS, SELECTORS, TILERS
from adaptile.config import load_config

KAYUAN = "configs/eval/kayuan.yaml"


def test_kayuan_config():
    cfg = load_config(KAYUAN)
    assert cfg.dataset.images == "data/kayuan/test/images"
    assert cfg.tiler == {"name": "uniform", "overlap": 0.2, "pad": False}
    assert cfg.tile_size == [640, 480, 320, 240, 160]
    assert list(cfg.selectors) == ["none", "gridnet"]
    assert cfg.selectors.gridnet == {"name": "gridnet", "threshold": 0.10,
                                     "weights": "weights/gridnet_kayuan_8x8.pth"}
    assert cfg.detector.name == "yolo" and cfg.detector.conf == 0.25
    assert cfg.merger == {"name": "nms", "iou_thr": 0.5, "max_det": None}
    assert cfg.out == "outputs/kayuan/gridnet_scaling_seed0.json"


def test_cli_overrides():
    cfg = load_config(KAYUAN, [
        "tile_selector.type=[none,tile_cnn]", "tile_selector.tile_cnn.threshold=0.3",
        "tiling.tile_size=640", "detector.conf=0.4", "merger.type=none",
        "eval.metrics=[fps]", "eval.max_images=50"])
    assert list(cfg.selectors) == ["none", "tile_cnn"]
    assert cfg.selectors.tile_cnn.threshold == 0.3
    assert cfg.tile_size == [640] and cfg.detector.conf == 0.4
    assert cfg.merger == {"name": "none"}
    assert cfg.metrics == ["fps"] and cfg.max_images == 50
    assert cfg.source.tile_selector.tile_cnn.threshold == 0.3     # bản lưu kèm kết quả


def test_no_tiling():
    cfg = load_config(KAYUAN, ["tiling.type=none", "tile_selector.type=none"])
    assert cfg.tiler == {"name": "none"}
    TILERS.build(cfg.tiler)


@pytest.mark.parametrize("override,msg", [
    ("tile_selector.type=yolo", "Không có selector 'yolo'"),
    ("tiling.type=random", "Không có tiler 'random'"),
    ("merger.type=wbf", "Không có merger 'wbf'"),
    ("eval.metrics=[map75]", "eval.metrics không hỗ trợ"),
])
def test_bad_values_fail_clearly(override, msg):
    with pytest.raises(KeyError, match=msg):
        load_config(KAYUAN, [override])


def test_selector_without_params_fails(tmp_path):
    src = open(KAYUAN).read().replace("  tile_cnn:", "  tile_cnn_cu:")
    f = tmp_path / "x.yaml"
    f.write_text(src)
    with pytest.raises(KeyError, match="thiếu mục tile_selector.tile_cnn"):
        load_config(str(f), ["tile_selector.type=[tile_cnn]"])


@pytest.mark.parametrize("path", ["configs/eval/kayuan.yaml", "configs/eval/seadronesee.yaml"])
def test_all_configs_load_and_build_cheap_parts(path):
    cfg = load_config(path)
    TILERS.build({**cfg.tiler, "tile_size": cfg.tile_size[0]})
    MERGERS.build(cfg.merger)
    SELECTORS.build(cfg.selectors.none)
