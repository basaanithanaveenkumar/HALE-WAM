"""MidBlock: bottleneck: ResBlock → Attention → ResBlock."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention_block import AttentionBlock
from .res_block import ResBlock


class MidBlock(nn.Module):
    """Bottleneck: ResBlock → Attention → ResBlock."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.res1 = ResBlock(ch, ch)
        self.attn = AttentionBlock(ch)
        self.res2 = ResBlock(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.attn(self.res1(x)))
