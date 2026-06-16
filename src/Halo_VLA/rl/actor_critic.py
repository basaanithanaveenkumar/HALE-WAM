"""
Actor, Critic, and the λ-return for Dreamer-style latent imagination.

The actor and critic both operate on the RSSM **feature** ``[h, z]`` — i.e.
purely in latent space.  They are trained on *imagined* trajectories rolled out
by the world model, never on real pixels, which is what makes Dreamer so
sample-efficient.

* **Actor** ``π(a | feature)`` — a diagonal Gaussian over actions.
* **Critic** ``V(feature)``     — predicts the expected λ-return from a state.
* **λ-return** — a low-variance value target that interpolates between a 1-step
  bootstrap (λ=0) and a full Monte-Carlo return (λ=1).
"""

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


class Critic(nn.Module):
    """State-value function ``V(feature)`` over the latent model state."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(-1)


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    continue_: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the TD(λ) / Dreamer λ-return target.

    Recurrence (Sutton & Barto; Dreamer):

        Gλ_t = r_t + γ·c_t · [ (1−λ)·V(s_{t+1}) + λ·Gλ_{t+1} ]

    with ``Gλ`` bootstrapped at the horizon by ``bootstrap`` (defaults to the
    last value).  λ=0 recovers the one-step TD target ``r + γ·c·V``; λ=1 recovers
    the discounted Monte-Carlo return.

    Args:
        reward:    [B, H] predicted rewards r_0..r_{H-1}.
        value:     [B, H] critic values V(s_0)..V(s_{H-1}).
        continue_: [B, H] continuation probabilities c_t ∈ [0, 1] (1 − done).
        gamma:     discount factor.
        lam:       the λ mixing coefficient.
        bootstrap: [B] value of the state *after* the horizon V(s_H); defaults to
                   ``value[:, -1]``.

    Returns:
        returns: [B, H] the λ-return target for each step.
    """
    B, H = reward.shape
    if bootstrap is None:
        bootstrap = value[:, -1]

    # next_values[t] = V(s_{t+1}); the last one uses the bootstrap value.
    next_values = torch.cat([value[:, 1:], bootstrap[:, None]], dim=1)   # [B, H]
    pcont = gamma * continue_                                             # [B, H]

    # inputs[t] = r_t + γ·c_t·(1−λ)·V(s_{t+1}); the λ·Gλ_{t+1} term is folded in
    # by the backward scan below.
    inputs = reward + pcont * next_values * (1.0 - lam)

    returns = torch.zeros_like(reward)
    agg = bootstrap                                  # Gλ_{H} = bootstrap
    for t in reversed(range(H)):
        agg = inputs[:, t] + pcont[:, t] * lam * agg
        returns[:, t] = agg
    return returns
