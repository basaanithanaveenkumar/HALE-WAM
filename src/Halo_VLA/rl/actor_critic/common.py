"""Helpers shared by the ``actor_critic`` package."""

from __future__ import annotations

import torch


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
    next_values = torch.cat([value[:, 1:], bootstrap[:, None]], dim=1)  # [B, H]
    pcont = gamma * continue_  # [B, H]

    # inputs[t] = r_t + γ·c_t·(1−λ)·V(s_{t+1}); the λ·Gλ_{t+1} term is folded in
    # by the backward scan below.
    inputs = reward + pcont * next_values * (1.0 - lam)

    returns = torch.zeros_like(reward)
    agg = bootstrap  # Gλ_{H} = bootstrap
    for t in reversed(range(H)):
        agg = inputs[:, t] + pcont[:, t] * lam * agg
        returns[:, t] = agg
    return returns
