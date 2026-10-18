"""PerceiverResampler: compress a variable token set to ``num_latents`` learned queries (Alayrac et al., 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Compress a variable token set to ``num_latents`` learned queries (Alayrac et al., 2022).

    RoboDual uses it to distil the observation features before the diffusion policy.
    """

    def __init__(self, dim: int, num_latents: int, heads: int) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compress observation tokens into ``num_latents`` tokens.

        Args:
            tokens: ``(B, N, dim)``.

        Returns:
            ``(B, num_latents, dim)``.
        """
        q = self.latents.expand(tokens.size(0), -1, -1)
        x = q + self.attn(q, self.norm(tokens), self.norm(tokens))[0]
        return x + self.ff(x)
