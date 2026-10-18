"""StubPatchEmbed: strided-convolution patch embedding for profiling."""

from __future__ import annotations

import torch
import torch.nn as nn


class StubPatchEmbed(nn.Module):
    """Embeds an image into ``num_patches`` tokens.

    Args:
        img_size: input resolution (square).
        p_size: patch size.
        in_chans: input channels.
        emb_dim: embedding width.
    """

    def __init__(
        self, img_size: int = 224, p_size: int = 16, in_chans: int = 3, emb_dim: int = 1024
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=p_size, stride=p_size)
        self.num_patches = (img_size // p_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``x`` ``[B, C, H, W]`` into patch tokens ``[B, num_patches, emb_dim]``."""
        return self.proj(x).flatten(2).transpose(1, 2)
