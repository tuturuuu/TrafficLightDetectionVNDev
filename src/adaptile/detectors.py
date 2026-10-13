"""Detector chạy trên các crop tile, trả detection theo tọa độ ảnh gốc."""

import sys
from contextlib import nullcontext

from .registry import DETECTORS
from .profiling import measure


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

    def __init__(self, weights, conf=0.25, imgsz=640, batch=16, device=None,
                 iou=None, max_det=None, half=None):
        register_custom_modules()
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.weights = str(weights)
        self.predict_options = {k: v for k, v in {'iou': iou, 'max_det': max_det, 'half': half}.items()
                                if v is not None}

    def prepare_shared_model(self):
        """Initialize the fine predictor once before sharing its fused backbone."""
        if self.model.predictor is None:
            options = {**self.model.overrides, 'conf': self.conf, 'imgsz': self.imgsz,
                       'batch': 1, 'save': False, 'mode': 'predict', 'rect': True,
                       **self.predict_options}
            if self.device is not None:
                options['device'] = self.device
            predictor = self.model._smart_load('predictor')(overrides=options, _callbacks=self.model.callbacks)
            predictor.setup_model(model=self.model.model, verbose=False)
            self.model.predictor = predictor
        backend_model = self.model.predictor.model.backend.model
        if backend_model is not self.model.model:
            raise RuntimeError('Coarse and fine detector must share the same native PyTorch model')
        return backend_model

    def __call__(self, crops, offsets):
        """crops: list ảnh BGR; offsets: list (x, y) góc trên-trái của crop.
        Trả list (x1, y1, x2, y2, score, cls) theo tọa độ ảnh gốc."""
        return self._detect(crops, offsets, profile=False)[0]

    def detect_with_details(self, crops, offsets):
        return self._detect(crops, offsets, profile=True)

    def _detect(self, crops, offsets, profile):
        dets, timings = [], {'fine_detector_ms': 0.0, 'remap_ms': 0.0}
        if len(crops) != len(offsets):
            raise ValueError('Crop and offset counts differ')
        kw = {**self.predict_options, **({} if self.device is None else {"device": self.device})}
        for i in range(0, len(crops), self.batch):
            chunk = crops[i:i + self.batch]
            offs = offsets[i:i + self.batch]
            device = next(self.model.model.parameters()).device
            with measure(timings, 'fine_detector_ms', device) if profile else nullcontext():
                results = self.model.predict(chunk, conf=self.conf,
                                             imgsz=self.imgsz, verbose=False, **kw)
            with measure(timings, 'remap_ms', device) if profile else nullcontext():
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
        return dets, timings
