"""Detector chạy trên các crop tile, trả detection theo tọa độ ảnh gốc."""

import sys

from .registry import DETECTORS


def register_custom_modules():
    """Đăng ký CBAM/SE với Ultralytics (một lần cho cả process).

    - `tasks.CBAM/SE`: để dựng model từ yaml (YOLOv8_CBAM.yaml, yolo26m-modified.yaml).
    - `sys.modules["custom_modules"]`: checkpoint cũ pickle class dưới tên này,
      alias về adaptile.models.attention để không phải chèn thư mục nào vào sys.path.
    """
    import ultralytics.nn.tasks as tasks
    from .models import attention

    sys.modules.setdefault("custom_modules", attention)
    tasks.CBAM = attention.CBAM
    tasks.SE = attention.SE


@DETECTORS.register("yolo")
class YOLODetector:
    """Một class cho mọi biến thể YOLO (v8, v8-CBAM, 26m-SE): khác nhau chỉ ở weights."""

    def __init__(self, weights, conf=0.25, imgsz=640, batch=16, device=None):
        register_custom_modules()
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.batch = batch
        self.device = device

    def __call__(self, crops, offsets):
        """crops: list ảnh BGR; offsets: list (x, y) góc trên-trái của crop.
        Trả list (x1, y1, x2, y2, score, cls) theo tọa độ ảnh gốc."""
        dets = []
        kw = {} if self.device is None else {"device": self.device}
        for i in range(0, len(crops), self.batch):
            chunk = crops[i:i + self.batch]
            offs = offsets[i:i + self.batch]
            results = self.model.predict(chunk, conf=self.conf,
                                         imgsz=self.imgsz, verbose=False, **kw)
            for res, (ox, oy) in zip(results, offs):
                if res.boxes is None:
                    continue
                b = res.boxes
                xyxy = b.xyxy.cpu().numpy()
                scores = b.conf.cpu().numpy()
                clss = b.cls.cpu().numpy()
                for (x1, y1, x2, y2), s, c in zip(xyxy, scores, clss):
                    dets.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                                 float(s), int(c)))
        return dets
