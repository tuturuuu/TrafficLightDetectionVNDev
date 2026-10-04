from ..registry import SELECTORS
from .base import AllTiles, TileSelector
from .gridnet import GridNetSelector
from .tile_cnn import TileCNNSelector

SELECTORS.register("none")(AllTiles)

__all__ = ["TileSelector", "AllTiles", "GridNetSelector", "TileCNNSelector"]
