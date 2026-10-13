"""Exact Eq. (7) learnable projection and Eq. (8) parameter-free comparator."""

import torch
from torch import nn


class LTSHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError('channels must be positive')
        self.proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        assert sum(p.numel() for p in self.parameters()) == channels + 1

    def forward(self, features):
        return self.proj(features)


def parameter_free_map(features):
    if features.ndim != 4:
        raise ValueError('Expected BCHW features')
    return torch.sigmoid(features).mean(dim=1, keepdim=True)
