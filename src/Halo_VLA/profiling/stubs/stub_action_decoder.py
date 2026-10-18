"""StubActionDecoder: MLP action head with the real decoder's shape, for profiling."""

from __future__ import annotations

import torch
import torch.nn as nn


class StubActionDecoder(nn.Module):
    """An MLP from a pooled decoder state to an ``(chunk_size, action_dim)`` action chunk.

    Args:
        emb_dim: input width.
        action_dim: action size.
        chunk_size: actions per chunk.
        hidden_dims: widths of the hidden layers.
    """

    def __init__(
        self,
        emb_dim: int = 1024,
        action_dim: int = 7,
        chunk_size: int = 8,
        hidden_dims: tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        dims = [emb_dim, *hidden_dims, action_dim * chunk_size]
        layers: list[nn.Module] = []
        for i, (di, do) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(di, do))
            if i < len(dims) - 2:
                layers.extend([nn.LayerNorm(do), nn.GELU()])
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict an action chunk.

        Args:
            x: pooled state ``[B, emb_dim]``.

        Returns:
            ``[B, chunk_size, action_dim]``.
        """
        return self.net(x).reshape(x.shape[0], self.chunk_size, self.action_dim)
