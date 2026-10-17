"""PooledLatent: multi-head attention pooling (MAP) of the last-layer states into one token (HiRT)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .latent_extractor import LatentExtractor


class PooledLatent(LatentExtractor):
    """Multi-head attention pooling (MAP) of the last-layer states into one token (HiRT)."""

    def __init__(self, width: int, heads: int = 8) -> None:
        super().__init__(width)
        heads = heads if width % heads == 0 else 1
        self.probe = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, enc):
        hidden, mask = enc["hidden"], enc["mask"]
        probe = self.probe.expand(hidden.size(0), -1, -1)
        pooled, _ = self.attn(probe, hidden, hidden, key_padding_mask=mask == 0)
        return self.norm(pooled), torch.ones(hidden.size(0), 1, device=hidden.device, dtype=mask.dtype)
