"""
Đọc file config của một thí nghiệm (vd configs/eval/kayuan.yaml).

Mỗi thí nghiệm là MỘT file, tự chứa đủ thông tin cho từng phần:

    dataset:        name, images, labels
    tiling:         type (uniform | none), tile_size (số hoặc danh sách), overlap, pad
    tile_selector:  type (none | tile_cnn | gridnet, một hoặc danh sách)
                    + một mục tham số cho mỗi selector có weight (gridnet:, tile_cnn:)
    detector:       type (yolo), weights, conf, imgsz, batch, device
    merger:         type (nms | none), iou_thr, max_det
    eval:           metrics, n_warmup, max_images, out

Ghi đè từ dòng lệnh bằng key có dấu chấm:
    tile_selector.type=[none,tile_cnn] tile_selector.gridnet.threshold=0.05 eval.max_images=50

load_config trả về dạng dùng nội bộ: cfg.tiler / cfg.detector / cfg.merger
(mỗi cái có `name` + tham số), cfg.selectors (dict tên -> config), cfg.tile_size
(danh sách), cfg.metrics, cfg.n_warmup, cfg.max_images, cfg.out, và cfg.source
(nguyên văn config sau khi ghi đè, để lưu kèm kết quả).
"""

from omegaconf import OmegaConf

from . import detectors, merge, selectors as _selectors, tiling  # noqa: F401  (đăng ký vào registry)
from .registry import DETECTORS, MERGERS, SELECTORS, TILERS

SECTIONS = ("dataset", "tiling", "tile_selector", "detector", "merger", "eval")
METRICS = ("map50", "map50_95", "precision", "recall", "agnostic", "fps")


def _as_list(v):
    return list(v) if OmegaConf.is_list(v) or isinstance(v, (list, tuple)) else [v]


def _component(section):
    """{type: x, a: 1, ...} -> {name: x, a: 1, ...}; type none thì bỏ tham số."""
    c = OmegaConf.to_container(section, resolve=True)
    c["name"] = c.pop("type")
    return {"name": "none"} if c["name"] == "none" else c


def load_config(path, overrides=None):
    src = OmegaConf.merge(OmegaConf.load(path),
                          OmegaConf.from_dotlist(list(overrides or [])))
    missing = [s for s in SECTIONS if s not in src]
    if missing:
        raise KeyError(f"{path}: thiếu mục {missing}")

    tiler = OmegaConf.to_container(src.tiling, resolve=True)
    tiler["name"] = tiler.pop("type")
    tile_size = _as_list(tiler.pop("tile_size", None))

    sel = OmegaConf.to_container(src.tile_selector, resolve=True)
    selectors = {}
    for name in _as_list(sel.pop("type")):
        SELECTORS.get(name)               # báo lỗi kèm danh sách nếu sai tên
        if name != "none" and name not in sel:
            raise KeyError(f"tile_selector.type có '{name}' nhưng thiếu mục "
                           f"tile_selector.{name} (weights, threshold, ...)")
        selectors[name] = {"name": name, **(sel.get(name) or {})}

    TILERS.get(tiler["name"])
    DETECTORS.get(src.detector.type)
    MERGERS.get(src.merger.type)

    ev = src.eval
    metrics = _as_list(ev.metrics)
    bad = set(metrics) - set(METRICS)
    if bad:
        raise KeyError(f"eval.metrics không hỗ trợ {sorted(bad)}. Có sẵn: {list(METRICS)}")

    return OmegaConf.create({
        "dataset": OmegaConf.to_container(src.dataset, resolve=True),
        "tiler": {"name": "none"} if tiler["name"] == "none" else tiler,
        "tile_size": tile_size,
        "selectors": selectors,
        "detector": _component(src.detector),
        "merger": _component(src.merger),
        "metrics": metrics,
        "n_warmup": ev.get("n_warmup", 3),
        "max_images": ev.get("max_images"),
        "out": ev.out,
        "source": src,
    })
