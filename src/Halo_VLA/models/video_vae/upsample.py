"""Upsample: 2× spatial upsampling via nearest-neighbour interpolation + convolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """2× spatial upsampling via nearest-neighbour interpolation + convolution."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
