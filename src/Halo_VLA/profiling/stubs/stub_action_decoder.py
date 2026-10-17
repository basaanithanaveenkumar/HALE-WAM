"""StubActionDecoder: stubActionDecoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class StubActionDecoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        action_dim: int = 7,
        chunk_size: int = 8,
        hidden_dims: tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        dims = [emb_dim, *hidden_dims, action_dim * chunk_size]
        layers: list[nn.Module] = []
        for i, (di, do) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(di, do))
            if i < len(dims) - 2:
                layers.extend([nn.LayerNorm(do), nn.GELU()])
        self.net       = nn.Sequential(*layers)
        self.action_dim   = action_dim
        self.chunk_size   = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(x.shape[0], self.chunk_size, self.action_dim)
