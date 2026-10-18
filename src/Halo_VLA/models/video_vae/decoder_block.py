"""DecoderBlock: num_res ResBlocks (first handles channel transition) + optional Upsample."""

from __future__ import annotations

import torch
import torch.nn as nn

from .res_block import ResBlock
from .upsample import Upsample


class DecoderBlock(nn.Module):
    """
    num_res ResBlocks (first handles channel transition) + optional Upsample.

    E.g. DecoderBlock(512, 256, num_res=3, upsample=True):
        ResBlock(512→256) → ResBlock(256→256) → ResBlock(256→256) → Upsample
    """

    def __init__(self, in_ch: int, out_ch: int, num_res: int = 3, upsample: bool = True) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ResBlock(in_ch, out_ch)]
        blocks += [ResBlock(out_ch, out_ch) for _ in range(num_res - 1)]
        self.resblocks = nn.Sequential(*blocks)
        self.up        = Upsample(out_ch) if upsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual blocks then the optional 2x upsampling.

        Args:
            x: ``[B, in_ch, H, W]``.

        Returns:
            ``[B, out_ch, H', W']`` with ``H' = 2H`` when ``upsample`` else ``H``.
        """
        return self.up(self.resblocks(x))
