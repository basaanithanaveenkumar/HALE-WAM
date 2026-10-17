"""StubPatchEmbed: stubPatchEmbed."""

from __future__ import annotations

import torch
import torch.nn as nn


class StubPatchEmbed(nn.Module):
    def __init__(self, img_size=224, p_size=16, in_chans=3, emb_dim=1024):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=p_size, stride=p_size)
        self.num_patches = (img_size // p_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)
