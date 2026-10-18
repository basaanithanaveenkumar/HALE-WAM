"""Critic: state-value function ``V(feature)`` over the latent model state."""

from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    """State-value function ``V(feature)`` over the latent model state.

    Args:
        feature_dim: width of the latent feature.
        hidden_dim: MLP hidden width.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Value estimates for ``feature`` ``(..., feature_dim)``: ``(...)``."""
        return self.net(feature).squeeze(-1)
