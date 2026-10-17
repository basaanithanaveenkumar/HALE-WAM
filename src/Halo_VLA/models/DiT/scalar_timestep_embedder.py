"""ScalarTimestepEmbedder: create sinusoidal embeddings from scalar timesteps t ∈ ℝ (e.g. flow time in [0, 1])."""

import math

import torch
import torch.nn as nn


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Create sinusoidal embeddings from scalar timesteps t ∈ ℝ (e.g. flow time in [0, 1]).
    t: [B] — each batch element gets an independent embedding row of size dim.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ScalarTimestepEmbedder(nn.Module):
    """Maps scalar t [B] → vector [B, hidden_size] (DiT / flow-matching compatible)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            t = t.reshape(t.size(0))
        t_freq = _timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
