"""PooledLatent: multi-head attention pooling (MAP) of the last-layer states into one token (HiRT)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig

from .latent_extractor import LatentExtractor
from .latent_registry import LATENT_EXTRACTORS


@LATENT_EXTRACTORS.register_module(name="pooled")
class PooledLatent(LatentExtractor):
    """Multi-head attention pooling (MAP) of the last-layer states into one token (HiRT).

    Args:
        width: token width.
        heads: attention heads of the pooling (falls back to 1 if ``width`` is not divisible).
    """

    def __init__(self, width: int, heads: int = 8) -> None:
        super().__init__(width)
        heads = heads if width % heads == 0 else 1
        self.probe = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    @classmethod
    def static_length(cls, config: DualSystemConfig) -> int | None:
        """The pooled latent is always one token."""
        return 1

    def forward(self, enc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention-pool all states into one token with a learned probe."""
        hidden, mask = enc["hidden"], enc["mask"]
        probe = self.probe.expand(hidden.size(0), -1, -1)
        pooled, _ = self.attn(probe, hidden, hidden, key_padding_mask=mask == 0)
        return self.norm(pooled), torch.ones(hidden.size(0), 1, device=hidden.device, dtype=mask.dtype)
