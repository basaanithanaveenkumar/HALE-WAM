"""QFormerLayer: pre-norm block: query self-attention, optional cross-attention to vision, FFN."""

from __future__ import annotations

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """Pre-norm block: query self-attention, optional cross-attention to vision, FFN."""

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        cross_attention: bool,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim) if cross_attention else None
        self.cross_attn = (
            nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, kdim=kv_dim, vdim=kv_dim, batch_first=True
            )
            if cross_attention
            else None
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    @property
    def has_cross_attention(self) -> bool:
        return self.cross_attn is not None

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.self_norm(queries)
        queries = queries + self.self_attn(h, h, h, need_weights=False)[0]
        if self.cross_attn is not None:
            h = self.cross_norm(queries)
            queries = queries + self.cross_attn(h, memory, memory, need_weights=False)[0]
        return queries + self.ffn(self.ffn_norm(queries))
