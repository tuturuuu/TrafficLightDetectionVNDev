"""
Hình học tile. Một tile là tuple (x1, y1, x2, y2, row, col) theo pixel ảnh gốc.

`make_grid` và `grid_mask_for_tiles` chuyển nguyên văn từ
legacy/dense_tiling_experiment.py; test/test_units.py kiểm tra chúng cho kết quả
giống hệt bản cũ.
"""

import cv2
import numpy as np

from .registry import TILERS

PAD_VALUE = (114, 114, 114)


def make_grid(H, W, tile_size, overlap=0.2):
    """Return list of (x1,y1,x2,y2, row, col) tiles covering the image."""
    stride = max(1, int(tile_size * (1 - overlap)))
    xs = list(range(0, max(W - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - tile_size, 0) + 1, stride)) or [0]
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    xs = sorted(set(max(0, x) for x in xs))
    ys = sorted(set(max(0, y) for y in ys))
    tiles = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            tiles.append((x, y, min(x + tile_size, W), min(y + tile_size, H),
                          r, c))
    return tiles, len(ys), len(xs)


def grid_mask_for_tiles(keep_mask, tiles, H, W, gr, gc):
    """OR-aggregate net cell decisions over each tile's real footprint."""
    out = np.zeros(len(tiles), dtype=bool)
    cell_h, cell_w = H / gr, W / gc
    for idx, (x1, y1, x2, y2, r, c) in enumerate(tiles):
        # net cell index range that this tile's pixel box overlaps
        r0 = int(y1 / cell_h); r1 = min(int((y2 - 1) / cell_h), gr - 1)
        c0 = int(x1 / cell_w); c1 = min(int((x2 - 1) / cell_w), gc - 1)
        out[idx] = keep_mask[r0:r1+1, c0:c1+1].any()
    return out


class Tiler:
    """Sinh tile và cắt crop. `pad=True` pad crop ở mép cho đủ tile_size
    (giống legacy/evaluations/evaluation_with_cnn.py); mặc định không pad, giống
    legacy/dense_tiling_experiment.py. Chỉ khác nhau khi tile lớn hơn cạnh ảnh."""

    pad = False
    tile_size = None

    def tiles(self, H, W):
        raise NotImplementedError

    def crop(self, image, tiles):
        crops = [image[y1:y2, x1:x2] for (x1, y1, x2, y2, _, _) in tiles]
        if self.pad and self.tile_size:
            crops = [_pad(c, self.tile_size) for c in crops]
        return crops


def _pad(crop, size):
    h, w = crop.shape[:2]
    if h >= size and w >= size:
        return crop
    return cv2.copyMakeBorder(crop, 0, max(size - h, 0), 0, max(size - w, 0),
                              cv2.BORDER_CONSTANT, value=PAD_VALUE)


@TILERS.register("uniform")
class UniformTiler(Tiler):
    def __init__(self, tile_size=640, overlap=0.2, pad=False):
        self.tile_size = tile_size
        self.overlap = overlap
        self.pad = pad

    def tiles(self, H, W):
        return make_grid(H, W, self.tile_size, self.overlap)[0]


@TILERS.register("none")
class NoTiler(Tiler):
    """Cả ảnh là một tile (YOLO chạy trên ảnh nguyên)."""

    def __init__(self):
        pass

    def tiles(self, H, W):
        return [(0, 0, W, H, 0, 0)]
