"""ActionTransformer: sinusoidal embedding of a continuous / integer timestep ``t`` (``(B,)`` -> ``(B, dim)``)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a continuous / integer timestep ``t`` (``(B,)`` -> ``(B, dim)``)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, dim % 2))


class ActionTransformer(nn.Module):
    """Action-chunk tokens that self-attend and cross-attend to the conditioning tokens.

    Shared by all three policy heads: for ``regression`` the input tokens are learned queries;
    for ``diffusion`` / ``flow_matching`` they are the embedded noisy actions plus a timestep
    embedding. Cross-attention to ``[latent | observation | proprioception]`` tokens is how the
    slow latent and the fast observation are fused (as in RoboDual's DiT and 3D Diffuser Actor).
    """

    def __init__(self, dim: int, layers: int, heads: int, chunk: int, action_dim: int) -> None:
        super().__init__()
        self.action_in = nn.Linear(action_dim, dim)
        self.queries = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, chunk, dim))
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, action_dim))
        self.dim = dim

    def forward(self, cond: torch.Tensor, cond_mask: torch.Tensor, noisy: torch.Tensor | None = None,
                t: torch.Tensor | None = None) -> torch.Tensor:
        b = cond.size(0)
        x = self.queries.expand(b, -1, -1) if noisy is None else self.action_in(noisy)
        if t is not None:
            x = x + self.time(timestep_embedding(t, self.dim)).unsqueeze(1)
        x = x + self.pos
        return self.out(self.decoder(x, cond, memory_key_padding_mask=cond_mask == 0))
