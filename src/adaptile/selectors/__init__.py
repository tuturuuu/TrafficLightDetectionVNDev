from ..registry import SELECTORS
from .base import AllTiles, TileSelector
from .gridnet import GridNetSelector
from .tile_cnn import TileCNNSelector
from .lts import LTSSelector
from .param_free_avg import ParameterFreeSelector

SELECTORS.register("none")(AllTiles)

__all__ = ["TileSelector", "AllTiles", "GridNetSelector", "TileCNNSelector",
           "LTSSelector", "ParameterFreeSelector"]
