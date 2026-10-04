"""
Pipeline: Tiler -> TileSelector -> Detector -> Merger.

    pipe = Pipeline.from_config(cfg)          # cfg có tiler/selector/detector/merger
    out = pipe(image_bgr)                     # ImageResult
    run = run_dataset(pipe, images)           # cả tập, có warmup + đo giờ

Quy tắc đo giờ giữ nguyên legacy/dense_tiling_experiment.py (xem RUNBOOK.md):
ảnh được đọc sẵn (imread không tính), 3 ảnh warmup không tính giờ,
torch.cuda.synchronize() trước mỗi lần đọc đồng hồ.
"""

import time
from dataclasses import dataclass, field

import torch

from . import selectors as _selectors  # noqa: F401  (đăng ký selector)
from .registry import DETECTORS, MERGERS, SELECTORS, TILERS
from .selectors.base import AllTiles


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class ImageResult:
    dets: list
    n_tiles: int
    n_kept: int
    t_select: float
    t_detect: float


@dataclass
class Pipeline:
    tiler: object
    selector: object
    detector: object
    merger: object

    @classmethod
    def from_config(cls, cfg, **overrides):
        """cfg: dict/DictConfig với 4 mục tiler, selector, detector, merger.
        `overrides` cho phép truyền sẵn object đã dựng (vd dùng chung detector)."""
        parts = {}
        for key, reg in (("tiler", TILERS), ("selector", SELECTORS),
                         ("detector", DETECTORS), ("merger", MERGERS)):
            parts[key] = overrides[key] if key in overrides else reg.build(cfg[key])
        return cls(**parts)

    def __call__(self, image):
        H, W = image.shape[:2]
        tiles = self.tiler.tiles(H, W)

        t_select = 0.0
        if isinstance(self.selector, AllTiles):
            tiles_run = tiles
        else:
            t0 = time.perf_counter()
            keep = self.selector.select(image, tiles)
            _sync()
            t_select = time.perf_counter() - t0
            tiles_run = [t for t, k in zip(tiles, keep) if k]
            if not tiles_run and self.selector.min_tiles:   # safety floor
                tiles_run = tiles[:self.selector.min_tiles]

        t0 = time.perf_counter()
        crops = self.tiler.crop(image, tiles_run)
        offsets = [(t[0], t[1]) for t in tiles_run]
        dets = self.detector(crops, offsets)
        _sync()
        t_detect = time.perf_counter() - t0

        return ImageResult(self.merger(dets), len(tiles), len(tiles_run),
                           t_select, t_detect)

    def warmup(self, images):
        """Giống warmup cũ: mỗi ảnh chạy selector + YOLO trên 2 tile đầu."""
        for img in images:
            tiles = self.tiler.tiles(*img.shape[:2])[:2]
            self.selector.warmup(img)
            self.detector(self.tiler.crop(img, tiles), [(t[0], t[1]) for t in tiles])
        _sync()


@dataclass
class DatasetRun:
    preds: list = field(default_factory=list)
    total_tiles: int = 0
    kept_tiles: int = 0
    select_s: float = 0.0
    detect_s: float = 0.0
    wall_s: float = 0.0


def run_dataset(pipe, images, n_warmup=3):
    """Chạy pipeline trên list ảnh (đã đọc sẵn). Trả DatasetRun."""
    pipe.warmup(images[:n_warmup])
    run = DatasetRun()
    t_wall0 = time.perf_counter()
    for img in images:
        out = pipe(img)
        run.preds.append(out.dets)
        run.total_tiles += out.n_tiles
        run.kept_tiles += out.n_kept
        run.select_s += out.t_select
        run.detect_s += out.t_detect
    _sync()
    run.wall_s = time.perf_counter() - t_wall0
    return run
