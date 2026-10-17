"""StubViT: stubViT."""

from __future__ import annotations

import torch
import torch.nn as nn

from .stub_patch_embed import StubPatchEmbed
from .stub_transformer_block import StubTransformerBlock


class StubViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        p_size: int = 16,
        emb_dim: int = 1024,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_dim: int = 2048,
    ):
        super().__init__()
        self.patch_emb = StubPatchEmbed(img_size, p_size, 3, emb_dim)
        n = self.patch_emb.num_patches
        self.pos_embed = nn.Embedding(n, emb_dim)
        self.blocks    = nn.ModuleList([
            StubTransformerBlock(emb_dim, n_heads, mlp_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        p = self.patch_emb(x)
        ids = torch.arange(p.shape[1], device=x.device).unsqueeze(0).expand(b, -1)
        p   = p + self.pos_embed(ids)
        for blk in self.blocks:
            p = blk(p)
        return self.norm(p)
