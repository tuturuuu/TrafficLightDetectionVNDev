import numpy as np


class TileSelector:
    """Quyết định tile nào được đưa qua detector.

    select(image, tiles) -> np.ndarray[bool] cùng độ dài với `tiles`.

    `min_tiles`: nếu không tile nào được chọn, pipeline vẫn chạy detector trên
    `min_tiles` tile đầu (giống legacy/dense_tiling_experiment.py). Đặt 0 để tắt.
    """

    min_tiles = 1

    def select(self, image, tiles):
        raise NotImplementedError

    def warmup(self, image):
        """Chạy thử một lần (không tính giờ). Mặc định không làm gì."""


class AllTiles(TileSelector):
    """Giữ mọi tile, tức uniform tiling."""

    def select(self, image, tiles):
        return np.ones(len(tiles), dtype=bool)
