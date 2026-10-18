"""Helpers shared by the ``rewards`` package."""

from __future__ import annotations

import torch


def negative_distance_reward(
    actions: torch.Tensor,
    target: torch.Tensor,
    squared: bool = True,
    offset: float = 0.0,
) -> torch.Tensor:
    """Reward = (offset) − distance(actions, target).

    The closer the action is to ``target``, the higher (less negative) the
    reward — a clean, convex, fully-deterministic signal that makes the unit
    tests reproducible.

    Args:
        actions: [B, A] sampled actions.
        target:  [A] or [B, A] goal action(s).
        squared: if True use squared-L2 (smooth); else L2 norm.
        offset:  constant added to every reward.  Adding a large constant is a
                 handy way to demonstrate that a *baseline* removes precisely
                 this offset and slashes the gradient variance.

    Returns:
        reward: [B].
    """
    if target.dim() == 1:
        target = target.unsqueeze(0)  # [1, A] broadcasts over batch
    diff = actions - target  # [B, A]
    dist_sq = diff.pow(2).sum(dim=-1)  # [B]
    dist = dist_sq if squared else dist_sq.clamp_min(1e-12).sqrt()
    return offset - dist
