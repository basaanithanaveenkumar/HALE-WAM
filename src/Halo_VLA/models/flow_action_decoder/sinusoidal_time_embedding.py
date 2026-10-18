"""SinusoidalTimeEmbedding: sinusoidal positional-style embedding for a continuous scalar t ∈ [0, 1]."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional-style embedding for a continuous scalar t ∈ [0, 1]."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Pre-compute log-spaced frequencies (not learnable)
        half = embed_dim // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / half))
        self.register_buffer("freq", freq)  # (half,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) or scalar → (batch, embed_dim)"""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # t: (batch,) → (batch, 1)
        t = t.unsqueeze(-1)  # (batch, 1)
        args = t * self.freq  # (batch, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, embed_dim)
