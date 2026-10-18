"""StubDecoderTransformer: bidirectional transformer stack with the real decoder's FLOPs, for profiling."""

from __future__ import annotations

import torch
import torch.nn as nn

from .stub_transformer_block import StubTransformerBlock


class StubDecoderTransformer(nn.Module):
    """Stacked :class:`StubTransformerBlock` layers followed by a LayerNorm and mean pooling.

    Args:
        emb_dim: model width.
        n_layers: number of blocks.
        n_heads: attention heads.
        mlp_dim: feed-forward hidden width.
    """

    def __init__(
        self, emb_dim: int = 1024, n_layers: int = 6, n_heads: int = 8, mlp_dim: int = 2048
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [StubTransformerBlock(emb_dim, n_heads, mlp_dim) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence and pool it.

        Args:
            x: ``[B, L, emb_dim]``.

        Returns:
            The mean over positions, ``[B, emb_dim]``.
        """
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).mean(dim=1)  # pool to [B, emb_dim]
