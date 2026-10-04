"""
Registry (factory) cho các thành phần của pipeline.

Mỗi loại thành phần (tiler, selector, detector, merger) có một Registry riêng.
Cài đặt mới chỉ cần đăng ký một lần:

    @SELECTORS.register("gridnet")
    class GridNetSelector(TileSelector): ...

rồi được dựng từ config:

    selector = SELECTORS.build({"name": "gridnet", "weights": "...", "threshold": 0.1})
"""

from collections.abc import Mapping


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, name):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' đã được đăng ký")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name):
        if name not in self._items:
            raise KeyError(f"Không có {self.kind} '{name}'. "
                           f"Có sẵn: {sorted(self._items)}")
        return self._items[name]

    def build(self, cfg, **extra):
        """cfg: dict / DictConfig có key `name`, các key còn lại là kwargs."""
        if not isinstance(cfg, Mapping):
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(cfg, resolve=True)
        kwargs = dict(cfg)
        name = kwargs.pop("name")
        return self.get(name)(**kwargs, **extra)

    def names(self):
        return sorted(self._items)


TILERS = Registry("tiler")
SELECTORS = Registry("selector")
DETECTORS = Registry("detector")
MERGERS = Registry("merger")
