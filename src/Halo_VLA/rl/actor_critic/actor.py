"""Actor: diagonal-Gaussian actor over actions, conditioned on the latent feature."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


class Actor(nn.Module):
    """Diagonal-Gaussian actor over actions, conditioned on the latent feature.

    Args:
        feature_dim: width of the latent feature.
        action_dim: action size.
        hidden_dim: MLP hidden width.
        log_std_min: lower clamp of the log standard deviation.
        log_std_max: upper clamp of the log standard deviation.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, feature: torch.Tensor) -> Independent:
        """The action distribution for ``feature`` ``(..., feature_dim)``: an independent Normal over ``action_dim``."""
        x = self.net(feature)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return Independent(Normal(mean, log_std.exp()), 1)

    def sample(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and its log-probability (no reparameterisation).

        Args:
            feature: latent feature ``(..., feature_dim)``.

        Returns:
            ``(action (..., action_dim), log_prob (...))``.
        """
        dist = self.distribution(feature)
        action = dist.sample()
        return action, dist.log_prob(action)

    def log_prob(self, feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-probability of ``action`` under the policy at ``feature``: ``(...)``."""
        return self.distribution(feature).log_prob(action)

    def entropy(self, feature: torch.Tensor) -> torch.Tensor:
        """Entropy of the policy at ``feature``: ``(...)``."""
        return self.distribution(feature).entropy()
