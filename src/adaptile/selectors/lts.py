import torch

from ..models.lts_checkpoint import load_lts_checkpoint
from ..registry import SELECTORS
from .relevance import RelevanceSelector


@SELECTORS.register('lts')
class LTSSelector(RelevanceSelector):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        self.checkpoint_metadata = None

    def bind_detector(self, detector):
        if self.bound_detector is detector:
            return
        super().bind_detector(detector)
        try:
            self.head, self.checkpoint_metadata = load_lts_checkpoint(
                self.weights, self.provider, detector.weights)
        except Exception:
            self.bound_detector = None
            raise

    def predict_map(self, features):
        return torch.sigmoid(self.head(features))
