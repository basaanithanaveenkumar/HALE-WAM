"""PatchEmb: non-overlapping image patches projected to embeddings."""

from __future__ import annotations

import torch
from torch import nn


class PatchEmb(nn.Module):
    """A strided convolution that turns an image into a sequence of patch embeddings.

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
        self.img_size = img_size
        self.p_size = p_size
        self.num_pat = (img_size // p_size) * (img_size // p_size)

        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=p_size, stride=p_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``x`` ``[B, C, H, W]`` into ``[B, num_pat, emb_dim]`` patch tokens."""
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
