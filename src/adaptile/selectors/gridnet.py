import torch

from ..models.gridnet import GridProposalNet
from ..registry import SELECTORS
from ..tiling import grid_mask_for_tiles
from .base import TileSelector


@SELECTORS.register("gridnet")
class GridNetSelector(TileSelector):
    """Một lượt GridNet trên cả ảnh -> lưới keep G×G -> giữ tile chạm ít nhất
    một ô "giữ" (grid_mask_for_tiles)."""

    def __init__(self, weights, threshold=0.2, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GridProposalNet.load(weights, device=self.device)
        self.threshold = threshold

    def select(self, image, tiles):
        H, W = image.shape[:2]
        keep, _ = self.net.propose(image, threshold=self.threshold)
        return grid_mask_for_tiles(keep, tiles, H, W,
                                   self.net.grid_rows, self.net.grid_cols)

    def warmup(self, image):
        self.net.propose(image, threshold=self.threshold)
