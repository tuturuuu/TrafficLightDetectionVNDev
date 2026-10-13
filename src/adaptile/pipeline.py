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
from contextlib import nullcontext

import torch
import numpy as np

from . import selectors as _selectors  # noqa: F401  (đăng ký selector)
from .registry import DETECTORS, MERGERS, SELECTORS, TILERS
from .selectors.base import AllTiles
from .profiling import measure


def _sync(*components):
    if torch.cuda.is_available():
        devices = set()
        for component in components:
            model = getattr(component, 'net', getattr(component, 'model', None))
            if model is not None:
                devices.update(p.device.index for p in model.parameters() if p.is_cuda)
        for device in devices if components else range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


@dataclass
class ImageResult:
    dets: list
    n_tiles: int
    n_kept: int
    t_select: float
    t_detect: float
    t_merge: float = 0.0
    t_total: float = 0.0
    selection: object = None
    kept_tiles: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


@dataclass
class Pipeline:
    tiler: object
    selector: object
    detector: object
    merger: object
    profile_stages: bool = False
    scoped_sync: bool = False

    def __post_init__(self):
        if hasattr(self.selector, 'bind_detector'):
            self.selector.bind_detector(self.detector)
            self.selector.profile_stages = self.profile_stages

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
        self.synchronize()
        t_start = time.perf_counter()
        H, W = image.shape[:2]
        tiles = self.tiler.tiles(H, W)
        selection = None
        timings = {}

        t_select = 0.0
        if isinstance(self.selector, AllTiles):
            tiles_run = tiles
        else:
            t0 = time.perf_counter()
            if hasattr(self.selector, 'select_with_details'):
                selection = self.selector.select_with_details(image, tiles)
                keep = selection.keep
                timings.update(selection.timings_ms)
            else:
                keep = self.selector.select(image, tiles)
            if np.asarray(keep).shape != (len(tiles),):
                raise ValueError('Selector must return one decision for every candidate tile')
            self.synchronize()
            t_select = time.perf_counter() - t0
            tiles_run = [t for t, k in zip(tiles, keep) if k]
            if not tiles_run and self.selector.min_tiles:   # safety floor
                tiles_run = tiles[:self.selector.min_tiles]

        t0 = time.perf_counter()
        with measure(timings, 'fine_crop_prepare_ms') if self.profile_stages else nullcontext():
            crops = self.tiler.crop(image, tiles_run)
            offsets = [(t[0], t[1]) for t in tiles_run]
        if self.profile_stages and hasattr(self.detector, 'detect_with_details'):
            dets, detail = self.detector.detect_with_details(crops, offsets)
            timings.update(detail)
        else:
            dets = self.detector(crops, offsets)
        self.synchronize()
        t_detect = time.perf_counter() - t0

        t0 = time.perf_counter()
        merged = self.merger(dets)
        self.synchronize()
        t_merge = time.perf_counter() - t0
        t_total = time.perf_counter() - t_start
        timings.update(nms_merge_ms=t_merge * 1000, end_to_end_ms=t_total * 1000)
        return ImageResult(merged, len(tiles), len(tiles_run),
                           t_select, t_detect, t_merge, t_total,
                           selection, tiles_run, timings)

    def warmup(self, images, full=False):
        """Giống warmup cũ: mỗi ảnh chạy selector + YOLO trên 2 tile đầu."""
        for img in images:
            if full:
                self(img)
                continue
            tiles = self.tiler.tiles(*img.shape[:2])[:2]
            self.selector.warmup(img)
            self.detector(self.tiler.crop(img, tiles), [(t[0], t[1]) for t in tiles])
        self.synchronize()

    def synchronize(self):
        if self.scoped_sync:
            _sync(self.detector, self.selector)
        else:
            _sync()


@dataclass
class DatasetRun:
    preds: list = field(default_factory=list)
    total_tiles: int = 0
    kept_tiles: int = 0
    select_s: float = 0.0
    detect_s: float = 0.0
    wall_s: float = 0.0
    merge_s: float = 0.0
    latency_s: dict = field(default_factory=lambda: {k: [] for k in ("select", "detect", "merge", "total")})
    peak_gpu_memory_mb: dict = field(default_factory=dict)
    image_results: list = field(default_factory=list)


def run_dataset(pipe, images, n_warmup=3, warmup_mode='legacy', retain_details=False):
    """Chạy pipeline trên list ảnh (đã đọc sẵn). Trả DatasetRun."""
    if warmup_mode not in ('legacy', 'full'):
        raise ValueError('warmup_mode must be legacy or full')
    if hasattr(pipe.selector, 'provider') and warmup_mode != 'full':
        raise ValueError('Shared-backbone selectors require warmup_mode=full')
    pipe.warmup(images[:n_warmup], full=warmup_mode == 'full')
    run = DatasetRun()
    pipe.synchronize()
    devices = []
    for component in (pipe.detector, pipe.selector):
        model = getattr(component, "net", getattr(component, "model", None))
        if model is not None:
            for parameter in model.parameters():
                if parameter.is_cuda and parameter.device.index not in devices:
                    devices.append(parameter.device.index)
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_wall0 = time.perf_counter()
    for img in images:
        out = pipe(img)
        run.preds.append(out.dets)
        if retain_details:
            run.image_results.append(out)
        run.total_tiles += out.n_tiles
        run.kept_tiles += out.n_kept
        run.select_s += out.t_select
        run.detect_s += out.t_detect
        run.merge_s += out.t_merge
        for stage in run.latency_s:
            run.latency_s[stage].append(getattr(out, "t_" + stage))
    pipe.synchronize()
    run.wall_s = time.perf_counter() - t_wall0
    run.peak_gpu_memory_mb = {str(d): torch.cuda.max_memory_allocated(d) / 2**20 for d in devices}
    return run
