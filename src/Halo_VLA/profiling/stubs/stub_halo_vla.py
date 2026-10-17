"""StubHaloVLA: full VLA stub: ViT → projector → decoder → action head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .stub_action_decoder import StubActionDecoder
from .stub_decoder_transformer import StubDecoderTransformer
from .stub_vit import StubViT


class StubHaloVLA(nn.Module):
    """Full VLA stub: ViT → projector → decoder → action head."""

    def __init__(
        self,
        emb_dim: int = 256,
        n_vit_layers: int = 3,
        n_dec_layers: int = 3,
        n_heads: int = 4,
        mlp_dim: int = 512,
    ):
        super().__init__()
        self.vit     = StubViT(emb_dim=emb_dim, n_layers=n_vit_layers, n_heads=n_heads, mlp_dim=mlp_dim)
        self.proj    = nn.Linear(emb_dim, emb_dim)
        self.decoder = StubDecoderTransformer(emb_dim=emb_dim, n_layers=n_dec_layers, n_heads=n_heads, mlp_dim=mlp_dim)
        self.action  = StubActionDecoder(emb_dim=emb_dim)

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis  = self.vit(images)          # (B, patches, emb)
        vis  = self.proj(vis)
        ctx  = torch.cat([vis, state.unsqueeze(1)], dim=1)
        h    = self.decoder(ctx)
        return self.action(h)
