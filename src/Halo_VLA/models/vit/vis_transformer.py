"""VisTransformer: the built-in Vision Transformer encoder."""

from __future__ import annotations

import torch
from torch import nn

from models.transformer import TransformerBlock

from .patch_emb import PatchEmb


class VisTransformer(nn.Module):
    """Vision Transformer: patch embedding, learned positions and bidirectional blocks.

    Args:
        img_size: input resolution (square).
        p_size: patch size.
        in_chans: input channels.
        emb_dim: embedding width.
        num_layers: transformer blocks.
        num_heads: attention heads.
        mlp_dim: feed-forward hidden width.
        drop_fact: dropout probability.
    """

    def __init__(
        self,
        img_size: int = 224,
        p_size: int = 16,
        in_chans: int = 3,
        emb_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        drop_fact: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_emb = PatchEmb(
            img_size=img_size, p_size=p_size, in_chans=in_chans, emb_dim=emb_dim
        )
        # learnable positional embeddings
        self.num_pat = (img_size // p_size) * (img_size // p_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_pat, emb_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(drop_fact)
        self.lay_norm = nn.LayerNorm(emb_dim)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    drop_fact=drop_fact,
                    causal_mask=False,
                    use_moe=False,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images ``[B, C, H, W]`` into patch tokens ``[B, num_pat, emb_dim]``."""
        x = self.patch_emb(x)  # [B, num_pat, emb_dim]
        x = x + self.pos_embed  # Add positional embeddings
        x = self.dropout(x)

        for block in self.transformer:
            x = block(x)  # Apply each TransformerBlock sequentially # [B, num_pat, emb_dim]
        x = self.lay_norm(x)
        return x
