"""Downsample: 2× spatial downsampling via strided convolution."""

from __future__ import annotations

import torch
import torch.nn as nn


class Downsample(nn.Module):
    """2× spatial downsampling via strided convolution."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
