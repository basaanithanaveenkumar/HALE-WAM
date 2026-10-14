"""
Building preference pairs for Direct Preference Optimization (DPO).

DPO learns from *comparisons* rather than scalar rewards: each training example
is a triple ``(state, chosen_action, rejected_action)`` where ``chosen`` is the
preferred of the two.  Preferences can come from a human, a learned reward, or —
as here — an analytic/world-model reward used to *label* which of two candidate
actions is better.

This module turns a batch of candidate actions + rewards into chosen/rejected
pairs.
"""

from __future__ import annotations

import torch

from rl.policy import GaussianActionPolicy


def build_preference_pairs(
    candidate_actions: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the best and worst candidate per state to form a preference pair.

    Args:
        candidate_actions: [B, K, A] — ``K`` candidate actions per state.
        rewards:           [B, K]    — reward of each candidate.

    Returns:
        chosen:   [B, A] — highest-reward candidate per state (the "winner").
        rejected: [B, A] — lowest-reward candidate per state (the "loser").
    """
    if candidate_actions.dim() != 3:
        raise ValueError("candidate_actions must be [B, K, A]")
    B = candidate_actions.size(0)
    best = rewards.argmax(dim=1)                       # [B]
    worst = rewards.argmin(dim=1)                      # [B]
    idx = torch.arange(B, device=candidate_actions.device)
    chosen = candidate_actions[idx, best]             # [B, A]
    rejected = candidate_actions[idx, worst]          # [B, A]
    return chosen, rejected


@torch.no_grad()
def sample_preference_pairs(
    policy: GaussianActionPolicy,
    cond: torch.Tensor,
    reward_fn,
    num_candidates: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``num_candidates`` actions per state from ``policy`` and label the
    best/worst by ``reward_fn`` to produce a preference pair.

    Args:
        policy:         policy used to propose candidate actions.
        cond:           [B, cond_dim] conditioning vectors.
        reward_fn:      callable ``actions[M, A] -> reward[M]``.
        num_candidates: candidates ``K`` sampled per state.

    Returns:
        (chosen, rejected), each [B, A].
    """
    B, _ = cond.shape
    K = num_candidates
    cond_rep = cond.repeat_interleave(K, dim=0)              # [B*K, D]
    actions = policy.distribution(cond_rep).sample()        # [B*K, A]
    rewards = reward_fn(actions).view(B, K)                 # [B, K]
    cand = actions.view(B, K, -1)                           # [B, K, A]
    return build_preference_pairs(cand, rewards)
