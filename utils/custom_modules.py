import torch
import torch.nn as nn


# -------------------------
# Spatial Attention
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        return self.sigmoid(self.conv(x))


# -------------------------
# Channel Attention (lazy init)
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, ratio=16):
        super().__init__()
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = None  # lazy initialization

    def _init_layers(self, c):
        hidden = max(c // self.ratio, 1)  # safety for small channels
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
        )

    def forward(self, x):
        if self.mlp is None:
            self._init_layers(x.shape[1])
        avg = self.mlp(self.avg_pool(x))
        max = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg + max)


# -------------------------
# CBAM (channel-agnostic)
# -------------------------
class CBAM(nn.Module):
    def __init__(self, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# -------------------------
# SE
# -------------------------
import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.r = r
        self.conv1 = None
        self.conv2 = None

    def forward(self, x):
        b, c, _, _ = x.shape

        # Lazy init (key fix)
        if self.conv1 is None:
            self.conv1 = nn.Conv2d(c, c // self.r, 1).to(x.device)
            self.conv2 = nn.Conv2d(c // self.r, c, 1).to(x.device)

        y = self.pool(x)
        y = self.conv2(torch.relu(self.conv1(y)))
        y = torch.sigmoid(y)

        return x * y