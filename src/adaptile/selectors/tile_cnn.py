import numpy as np
import torch
import torch.nn.functional as F

from ..models.tile_cnn import IMG_SIZE, TileProposalCNN, load_state_dict
from ..registry import SELECTORS
from ..tiling import _pad
from .base import TileSelector


@SELECTORS.register("tile_cnn")
class TileCNNSelector(TileSelector):
    """CNN chấm điểm từng tile (thế hệ 1). Chi phí tăng theo số tile.

    Tiền xử lý giống legacy/evaluations/evaluation_with_cnn.py: crop được pad (114)
    cho vuông, BGR->RGB, resize bilinear về 160 trên GPU, /255; giữ tile có
    sigmoid > threshold. Mọi tile của một ảnh nằm chung batch (tối đa
    `batch_size`), không trộn tile của nhiều ảnh.
    """

    min_tiles = 0     # code cũ không có safety floor cho CNN

    def __init__(self, weights, threshold=0.5, batch_size=64, device=None):
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.net = TileProposalCNN().to(self.device)
        load_state_dict(self.net, weights, self.device)
        self.net.eval()
        self.threshold = threshold
        self.batch_size = batch_size

    @torch.no_grad()
    def scores(self, image, tiles):
        size = max(max(x2 - x1, y2 - y1) for (x1, y1, x2, y2, _, _) in tiles)
        crops = [_pad(image[y1:y2, x1:x2], size)
                 for (x1, y1, x2, y2, _, _) in tiles]
        out = []
        for i in range(0, len(crops), self.batch_size):
            batch = torch.from_numpy(np.stack(crops[i:i + self.batch_size]))
            batch = batch.to(self.device).permute(0, 3, 1, 2).contiguous().flip(1)
            if batch.shape[2] != IMG_SIZE or batch.shape[3] != IMG_SIZE:
                batch = F.interpolate(batch.float(), size=(IMG_SIZE, IMG_SIZE),
                                      mode="bilinear", align_corners=False)
            else:
                batch = batch.float()
            logits = self.net(batch / 255.0).squeeze(1)
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)

    def select(self, image, tiles):
        return self.scores(image, tiles) > self.threshold

    def warmup(self, image):
        H, W = image.shape[:2]
        self.scores(image, [(0, 0, min(W, 640), min(H, 640), 0, 0)])
