"""
Unit tests for GRPO (branch ``rl/grpo``).

Run with:
    pytest tests/test_grpo.py -o addopts="" -q
or standalone:
    python tests/test_grpo.py
"""

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from rl.policy import GaussianActionPolicy
from rl.grpo import compute_grpo_loss, group_relative_advantage
from rl.rewards import negative_distance_reward


def _fresh_ref(policy):
    """A frozen deep copy: identical params ⇒ KL=0 and ratio=1 at the first step."""
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def test_group_relative_advantage_is_standardised():
    """Each group's advantages are zero-mean and (for G>1) unit-std."""
    torch.manual_seed(0)
    rewards = torch.randn(5, 8) * 3 + 7   # arbitrary mean/scale per group
    adv = group_relative_advantage(rewards)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(adv.std(dim=1, unbiased=False), torch.ones(5), atol=1e-3)
    print("test_group_relative_advantage_is_standardised passed.")


def test_grpo_loss_scalar_and_backprops():
    torch.manual_seed(0)
    N, cond_dim, A = 4, 16, 6
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    ref = _fresh_ref(policy)
    cond = torch.randn(N, cond_dim)
    target = torch.randn(A)

    def reward_fn(a, g=8):
        return negative_distance_reward(a, target)

    loss, metrics = compute_grpo_loss(
        policy, ref, cond, reward_fn, group_size=8, kl_beta=0.04
    )
    assert loss.dim() == 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    print("test_grpo_loss_scalar_and_backprops passed.")


def test_grpo_at_init_kl_zero_and_loss_near_zero():
    """With ref == policy: ratio≈1, KL≈0, and surrogate≈mean(Â)≈0 ⇒ loss≈0."""
    torch.manual_seed(0)
    N, cond_dim, A = 6, 16, 4
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    ref = _fresh_ref(policy)
    cond = torch.randn(N, cond_dim)
    target = torch.randn(A)

    loss, metrics = compute_grpo_loss(
        policy, ref, cond, lambda a: negative_distance_reward(a, target),
        group_size=16, kl_beta=0.04,
    )
    assert abs(metrics["kl"]) < 1e-5, metrics["kl"]
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-5, metrics["ratio_mean"]
    # group-normalised advantage is mean-zero per group ⇒ surrogate ≈ 0.
    assert abs(metrics["surrogate"]) < 1e-4, metrics["surrogate"]
    assert abs(metrics["loss"]) < 1e-4, metrics["loss"]
    print("test_grpo_at_init_kl_zero_and_loss_near_zero passed.")


def test_grpo_increases_preference_margin():
    """GRPO widens the log-prob margin between the best and worst group sample.

    Because the group-relative advantage is positive for above-average actions
    and negative for below-average ones, the clipped surrogate pushes π_θ to put
    *more* mass on the best action and *less* on the worst.  We track the margin
    ``log π(best) − log π(worst)`` on a fixed set of actions across a few
    optimiser steps; it should increase.  (The absolute log-probs can drift
    because σ is shared across all states, but their difference cancels the
    shared terms and is the clean quantity to test.)
    """
    torch.manual_seed(0)
    N, cond_dim, A, G = 4, 16, 4, 16
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    ref = _fresh_ref(policy)
    cond = torch.randn(N, cond_dim)
    target = torch.zeros(A)

    # Freeze a group of actions so we compare like-for-like across updates.
    cond_rep = cond.repeat_interleave(G, dim=0)
    with torch.no_grad():
        actions = ref.distribution(cond_rep).sample()           # [N*G, A]
        rewards = negative_distance_reward(actions, target)     # [N*G]
    rg = rewards.view(N, G)
    best_idx = rg.argmax(dim=1) + torch.arange(N) * G
    worst_idx = rg.argmin(dim=1) + torch.arange(N) * G

    def margin():
        lp = policy.log_prob(cond_rep, actions)
        return (lp[best_idx] - lp[worst_idx]).mean().item()

    margin_before = margin()

    # A few GRPO updates on these exact actions (kl_beta=0 isolates the surrogate).
    opt = torch.optim.Adam(policy.parameters(), lr=0.02)
    adv = group_relative_advantage(rg).view(N * G)
    logp_old = ref.log_prob(cond_rep, actions).detach()
    for _ in range(30):
        logp_new = policy.log_prob(cond_rep, actions)
        ratio = torch.exp(logp_new - logp_old)
        surrogate = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
        opt.zero_grad()
        (-surrogate).backward()
        opt.step()

    margin_after = margin()
    assert margin_after > margin_before, (
        f"preference margin should grow: before={margin_before:.3f} "
        f"after={margin_after:.3f}"
    )
    print(f"test_grpo_increases_preference_margin passed "
          f"({margin_before:.3f} -> {margin_after:.3f}).")


if __name__ == "__main__":
    test_group_relative_advantage_is_standardised()
    test_grpo_loss_scalar_and_backprops()
    test_grpo_at_init_kl_zero_and_loss_near_zero()
    test_grpo_increases_preference_margin()
    print("\nAll GRPO tests passed.")
