"""GatedCrossAttentionBlock: one tanh-gated cross-attention + tanh-gated FFN residual block."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedCrossAttentionBlock(nn.Module):
    """One tanh-gated cross-attention + tanh-gated FFN residual block.

    Args:
        dim: latent width.
        kv_dim: vision-token width.
        num_heads: attention heads.
        ffn_mult: feed-forward expansion factor.
        dropout: dropout probability.
        gate_init: initial gate value before ``tanh``.
    """

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, kdim=kv_dim, vdim=kv_dim, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )
        self.attn_gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.ffn_gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Blend the vision memory into the latents through the gated residuals.

        Args:
            x: latents ``[B, L, dim]``.
            memory: vision tokens ``[B, N, kv_dim]``.

        Returns:
            ``[B, L, dim]``.
        """
        h = self.attn_norm(x)
        x = x + torch.tanh(self.attn_gate) * self.attn(h, memory, memory, need_weights=False)[0]
        return x + torch.tanh(self.ffn_gate) * self.ffn(self.ffn_norm(x))
