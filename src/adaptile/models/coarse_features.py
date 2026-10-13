"""One graph-aware backbone pass through the SAME detector used for fine crops."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from ..profiling import measure
from ..provenance import architecture_sha256, software_versions


@dataclass(frozen=True)
class CoarseGeometry:
    original_hw: tuple
    input_hw: tuple
    resized_hw: tuple
    pad_xy: tuple

    @property
    def scale_xy(self):
        return (self.resized_hw[1] / self.original_hw[1],
                self.resized_hw[0] / self.original_hw[0])

    def transform_boxes(self, boxes):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, self.original_hw[1])
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, self.original_hw[0])
        boxes *= np.tile(self.scale_xy, 2)
        boxes += np.tile(self.pad_xy, 2)
        return boxes


def letterbox(image, size):
    """Fixed square, centered YOLO letterbox; record actual rounded pixel geometry."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR HWC image')
    h, w = image.shape[:2]
    if min(h, w) <= 0:
        raise ValueError('Empty image')
    ratio = min(size / h, size / w)
    rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
    top, left = round((size - rh) / 2 - .1), round((size - rw) / 2 - .1)
    bottom, right = size - rh - top, size - rw - left
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rh, rw) != (h, w) else image
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    return rgb, CoarseGeometry((h, w), (size, size), (rh, rw), (left, top))


@contextmanager
def evaluating(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


@dataclass
class FeatureBatch:
    features: torch.Tensor
    geometries: list
    timings_ms: dict


class YOLOCoarseFeatureProvider:
    def __init__(self, detector, feature_stage, coarse_imgsz=640):
        # A raw DetectionModel is accepted for training; no inference backend/fusion there.
        self.detector = detector
        self.model = detector.prepare_shared_model() if hasattr(detector, 'prepare_shared_model') else detector
        if not hasattr(self.model, 'yaml') or not hasattr(self.model, 'model'):
            raise TypeError('Expected an existing Ultralytics DetectionModel or YOLODetector')
        if isinstance(feature_stage, bool) or not isinstance(feature_stage, int):
            raise ValueError('feature_stage must be an explicit backbone layer index')
        if not 0 <= feature_stage < len(self.model.yaml['backbone']):
            raise ValueError('feature_stage must identify a backbone layer, not neck or Detect')
        if not isinstance(coarse_imgsz, int) or coarse_imgsz < 32 or coarse_imgsz % 32:
            raise ValueError('coarse_imgsz must be a positive multiple of 32')
        self.feature_stage = feature_stage
        self.coarse_imgsz = coarse_imgsz
        parameter = next(self.model.parameters())
        with evaluating(self.model), torch.no_grad():
            probe = self.forward_tensor(torch.zeros(1, 3, 64, 64, device=parameter.device, dtype=parameter.dtype))
        self.channels = probe.shape[1]
        self.stride_hw = (64 / probe.shape[2], 64 / probe.shape[3])
        self.metadata = {
            'architecture_sha256': architecture_sha256(self.model),
            'feature_stage': feature_stage, 'channels': self.channels,
            'stride_hw': list(self.stride_hw), 'coarse_imgsz': coarse_imgsz,
            'feature_shape_bchw': [None, self.channels,
                                   int(coarse_imgsz / self.stride_hw[0]),
                                   int(coarse_imgsz / self.stride_hw[1])],
            'dtype': str(parameter.dtype), 'device': str(parameter.device),
            'preprocessing': 'letterbox_rgb_0_1_actual_rounded_scale_v1',
            'versions': software_versions(),
        }

    def forward_tensor(self, tensor):
        """Honor YOLO graph dependencies and stop before all later backbone/head layers."""
        saved = []
        x = tensor
        for i, layer in enumerate(self.model.model):
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            if i == self.feature_stage:
                if not torch.is_tensor(x) or x.ndim != 4:
                    raise ValueError('Configured stage must return one BCHW tensor')
                return x
            saved.append(x if i in self.model.save else None)
        raise RuntimeError('Feature stage was not reached')

    def __call__(self, images, grad_enabled=False, profile=True):
        images = [images] if isinstance(images, np.ndarray) else list(images)
        if not images:
            raise ValueError('Coarse batch cannot be empty')
        parameter = next(self.model.parameters())
        times = {}
        with measure(times, 'coarse_preprocess_ms', parameter.device) if profile else nullcontext():
            arrays, geometry = zip(*(letterbox(image, self.coarse_imgsz) for image in images))
            tensor = torch.from_numpy(np.stack(arrays)).to(device=parameter.device, dtype=parameter.dtype) / 255
        with measure(times, 'coarse_backbone_ms', parameter.device) if profile else nullcontext():
            # In joint mode the trainer owns module modes; gradients must reach this model.
            if grad_enabled:
                features = self.forward_tensor(tensor)
            else:
                with evaluating(self.model), torch.no_grad():
                    features = self.forward_tensor(tensor)
        return FeatureBatch(features, list(geometry), times)
