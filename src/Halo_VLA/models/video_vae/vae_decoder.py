"""VAEDecoder: [B, latent_ch, H/8, W/8] → [B, 3, H, W]."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_block import DecoderBlock
from .mid_block import MidBlock


class VAEDecoder(nn.Module):
    """
    [B, latent_ch, H/8, W/8] → [B, 3, H, W]

    Channel schedule : [512, 512, 256, 128]  (mirror of encoder)
    Upsampling       : ×2 after blocks 0, 1, 2  →  8× total
    Bottleneck       : ResBlock + Attention + ResBlock at 1/8 resolution
    """

    def __init__(self, latent_ch: int = 4, out_ch: int = 3, base_ch: int = 128) -> None:
        super().__init__()
        ch = [base_ch * 4, base_ch * 4, base_ch * 2, base_ch]

        self.conv_in = nn.Conv2d(latent_ch, ch[0], 3, padding=1)
        self.mid     = MidBlock(ch[0])

        self.blocks = nn.ModuleList([
            DecoderBlock(ch[0], ch[1], num_res=3, upsample=True),    # H/8 → H/4
            DecoderBlock(ch[1], ch[2], num_res=3, upsample=True),    # H/4 → H/2
            DecoderBlock(ch[2], ch[3], num_res=3, upsample=True),    # H/2 → H
            DecoderBlock(ch[3], ch[3], num_res=3, upsample=False),   # H   (final refine)
        ])

        self.norm_out = nn.GroupNorm(32, ch[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(ch[-1], out_ch, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.

        Args:
            z: latent sample ``[B, latent_ch, H/8, W/8]``.

        Returns:
            Reconstructed images ``[B, out_ch, H, W]``.
        """
        h = self.mid(self.conv_in(z))
        for block in self.blocks:
            h = block(h)
        return self.conv_out(F.silu(self.norm_out(h)))  # [B, out_ch, H, W]
