"""StubTransformerBlock: pre-norm attention + MLP block for profiling."""

from __future__ import annotations

import torch
import torch.nn as nn


class StubTransformerBlock(nn.Module):
    """A standard pre-norm transformer block with the real model's operator mix.

    Args:
        emb_dim: model width.
        n_heads: attention heads.
        mlp_dim: feed-forward hidden width.
    """

    def __init__(self, emb_dim: int, n_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff    = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention then the MLP, each residual: ``[B, L, emb_dim] -> [B, L, emb_dim]``."""
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x
