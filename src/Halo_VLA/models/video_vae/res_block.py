"""ResBlock: pre-activation residual block: GroupNorm → SiLU → Conv, ×2, + skip."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-activation residual block: GroupNorm → SiLU → Conv, ×2, + skip."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward: ``[B, in_ch, H, W] -> [B, out_ch, H, W]``."""
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
