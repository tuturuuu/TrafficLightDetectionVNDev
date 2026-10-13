"""GT-overlap targets and padding-masked relevance BCE (Eq. 10)."""

import numpy as np
import torch
import torch.nn.functional as F

from .selectors.relevance_utils import feature_bounds


def relevance_targets(boxes_per_image, geometries, feature_hw, device=None):
    if len(boxes_per_image) != len(geometries):
        raise ValueError('GT and geometry batch lengths differ')
    fh, fw = feature_hw
    target = np.zeros((len(geometries), 1, fh, fw), np.float32)
    valid = np.zeros_like(target)
    for i, (boxes, geometry) in enumerate(zip(boxes_per_image, geometries)):
        h, w = geometry.original_hw
        content, _ = feature_bounds((0, 0, w, h), geometry, feature_hw)
        valid[i, 0][content] = 1  # Include partially covered cells, exclude padding-only cells.
        for box in boxes:
            if not np.isfinite(box[:4]).all():
                raise ValueError('GT coordinates must be finite')
            bounds, _ = feature_bounds(box[:4], geometry, feature_hw)
            if bounds is not None:
                target[i, 0][bounds] = 1
    return torch.from_numpy(target).to(device), torch.from_numpy(valid).to(device)


def relevance_loss(logits, targets, valid, pos_weight=1.0):
    if logits.shape != targets.shape or logits.shape != valid.shape or logits.ndim != 4:
        raise ValueError('Logits, targets and mask must share B1HW shape')
    if logits.shape[1] != 1 or not np.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError('Expected one relevance channel and positive pos_weight')
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none',
        pos_weight=torch.as_tensor(pos_weight, device=logits.device, dtype=torch.float32))
    return (loss * valid).sum() / valid.sum().clamp_min(1)
