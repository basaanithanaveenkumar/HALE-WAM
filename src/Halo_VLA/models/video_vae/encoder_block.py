"""EncoderBlock: num_res ResBlocks (first handles channel transition) + optional Downsample."""

from __future__ import annotations

import torch
import torch.nn as nn

from .downsample import Downsample
from .res_block import ResBlock


class EncoderBlock(nn.Module):
    """
    num_res ResBlocks (first handles channel transition) + optional Downsample.

    E.g. EncoderBlock(128, 256, num_res=2, downsample=True):
        ResBlock(128→256) → ResBlock(256→256) → Downsample
    """

    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2, downsample: bool = True) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResBlock(in_ch, out_ch)]
        blocks += [ResBlock(out_ch, out_ch) for _ in range(num_res - 1)]
        self.resblocks = nn.Sequential(*blocks)
        self.down      = Downsample(out_ch) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.resblocks(x))
