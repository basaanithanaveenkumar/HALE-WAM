"""ImageProjector: the MLP vision -> decoder projector (one token per ViT patch)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageProjector(nn.Module):
    """A 3-layer MLP that maps vision features to the decoder width.

    Args:
        vision_dim: vision feature width.
        llm_dim: decoder width.
    """

    def __init__(self, vision_dim: int = 512, llm_dim: int = 4096) -> None:
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(vision_dim, llm_dim // 4),
                nn.LayerNorm(llm_dim // 4),
                nn.GELU(),
                nn.Linear(llm_dim//4, llm_dim // 2),
                nn.LayerNorm(llm_dim // 2),
                nn.GELU(),
                nn.Linear(llm_dim//2,llm_dim )
            )

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        """Project ``(B, N, vision_dim)`` features to ``(B, N, llm_dim)``."""
        return self.proj(img_features)
