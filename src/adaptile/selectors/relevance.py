"""Common shared-backbone selection geometry for Eq. (7) and Eq. (8)."""

from contextlib import nullcontext

import torch

from .base import TileSelector
from .relevance_utils import SelectionResult, score_tiles, select_scores
from ..models.coarse_features import YOLOCoarseFeatureProvider
from ..profiling import measure


class RelevanceSelector(TileSelector):
    min_tiles = 0

    def __init__(self, feature_stage, coarse_imgsz=640, threshold=.5, min_keep=0,
                 force_keep_all=False):
        select_scores([], threshold, min_keep)
        self.feature_stage = feature_stage
        self.coarse_imgsz = coarse_imgsz
        self.threshold = threshold
        self.min_keep = min_keep
        self.force_keep_all = force_keep_all
        self.provider = None
        self.bound_detector = None
        self.head = None
        self.profile_stages = True

    def bind_detector(self, detector):
        if self.bound_detector is detector:
            return
        self.provider = YOLOCoarseFeatureProvider(detector, self.feature_stage, self.coarse_imgsz)
        self.bound_detector = detector

    def predict_map(self, features):
        raise NotImplementedError

    def select_with_details(self, image, tiles):
        if self.provider is None:
            raise RuntimeError('Bind this selector to the fine detector before selection')
        batch = self.provider(image, profile=self.profile_stages)
        timings = dict(batch.timings_ms)
        timer = measure(timings, 'relevance_predict_ms', batch.features.device) if self.profile_stages else nullcontext()
        with timer, torch.no_grad():
            relevance = self.predict_map(batch.features)[0, 0].float().cpu().numpy()
        with measure(timings, 'candidate_tile_scoring_ms') if self.profile_stages else nullcontext():
            scores, fallbacks = score_tiles(relevance, tiles, batch.geometries[0])
            keep = select_scores(scores, self.threshold, self.min_keep, self.force_keep_all)
        return SelectionResult(keep, scores, relevance, timings, batch.geometries[0], fallbacks)

    def select(self, image, tiles):
        return self.select_with_details(image, tiles).keep

    def warmup(self, image):
        h, w = image.shape[:2]
        self.select(image, [(0, 0, w, h, 0, 0)])
