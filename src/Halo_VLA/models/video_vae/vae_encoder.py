"""VAEEncoder: [B, 3, H, W] → [B, latent_ch*2, H/8, W/8]   (mean ‖ logvar concatenated)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_block import EncoderBlock
from .mid_block import MidBlock


class VAEEncoder(nn.Module):
    """
    [B, 3, H, W] → [B, latent_ch*2, H/8, W/8]   (mean ‖ logvar concatenated)

    Channel schedule : [128, 256, 512, 512]
    Downsampling     : ×2 after blocks 0, 1, 2  →  8× total
    Bottleneck       : ResBlock + Attention + ResBlock at 1/8 resolution
    """

    def __init__(self, in_ch: int = 3, latent_ch: int = 4, base_ch: int = 128) -> None:
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]

        self.conv_in = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        self.blocks = nn.ModuleList([
            EncoderBlock(ch[0], ch[0], num_res=2, downsample=True),   # H   → H/2
            EncoderBlock(ch[0], ch[1], num_res=2, downsample=True),   # H/2 → H/4
            EncoderBlock(ch[1], ch[2], num_res=2, downsample=True),   # H/4 → H/8
            EncoderBlock(ch[2], ch[3], num_res=2, downsample=False),  # H/8 (bottleneck)
        ])

        self.mid      = MidBlock(ch[-1])
        self.norm_out = nn.GroupNorm(32, ch[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(ch[-1], latent_ch * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.mid(h)
        return self.conv_out(F.silu(self.norm_out(h)))  # [B, latent_ch*2, H/8, W/8]
