"""Actor: diagonal-Gaussian actor over actions, conditioned on the latent feature."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


class Actor(nn.Module):
    """Diagonal-Gaussian actor over actions, conditioned on the latent feature."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, feature: torch.Tensor) -> Independent:
        x = self.net(feature)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return Independent(Normal(mean, log_std.exp()), 1)

    def sample(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(feature)
        action = dist.sample()
        return action, dist.log_prob(action)

    def log_prob(self, feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(feature).log_prob(action)

    def entropy(self, feature: torch.Tensor) -> torch.Tensor:
        return self.distribution(feature).entropy()
