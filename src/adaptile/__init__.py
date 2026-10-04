"""adaptile: YOLO + tiling + chọn tile (GridNet), ghép bằng config.

Dataset -> Tiler -> TileSelector -> Detector -> Merger -> Evaluator
"""

from . import detectors, merge, selectors, tiling  # noqa: F401  (đăng ký vào registry)
from .metrics import evaluate
from .pipeline import Pipeline, run_dataset
from .registry import DETECTORS, MERGERS, SELECTORS, TILERS

__all__ = ["Pipeline", "run_dataset", "evaluate",
           "TILERS", "SELECTORS", "DETECTORS", "MERGERS"]
