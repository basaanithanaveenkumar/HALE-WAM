"""
Unit tests for DPO on action sequences (branch ``rl/dpo``).

Run with:
    pytest tests/test_dpo.py -o addopts="" -q
or standalone:
    python tests/test_dpo.py
"""

import copy
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from rl.policy import GaussianActionPolicy
from rl.dpo import compute_dpo_loss
from rl.preference import build_preference_pairs, sample_preference_pairs
from rl.rewards import negative_distance_reward


def _fresh_ref(policy):
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def test_build_preference_pairs_picks_best_and_worst():
    """Highest-reward candidate becomes chosen; lowest becomes rejected."""
    cand = torch.tensor([[[0.0], [1.0], [2.0]]])      # [B=1, K=3, A=1]
    rewards = torch.tensor([[0.5, 9.0, -3.0]])        # best=idx1, worst=idx2
    chosen, rejected = build_preference_pairs(cand, rewards)
    assert torch.allclose(chosen, torch.tensor([[1.0]]))
    assert torch.allclose(rejected, torch.tensor([[2.0]]))
    print("test_build_preference_pairs_picks_best_and_worst passed.")


def test_dpo_loss_equals_log2_when_policy_is_reference():
    """If π_θ == π_ref, every logit is 0 and the loss is exactly log 2."""
    torch.manual_seed(0)
    B, cond_dim, A = 5, 16, 4
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    ref = _fresh_ref(policy)
    cond = torch.randn(B, cond_dim)
    chosen = torch.randn(B, A)
    rejected = torch.randn(B, A)

    loss, metrics = compute_dpo_loss(policy, ref, cond, chosen, rejected, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-5, loss.item()
    # implicit rewards are identical ⇒ zero margin, 0%/100% boundary accuracy.
    assert abs(metrics["margin"]) < 1e-6
    print("test_dpo_loss_equals_log2_when_policy_is_reference passed.")


def test_dpo_loss_scalar_and_backprops_policy_only():
    """Loss is scalar, gradients reach π_θ but never the frozen reference."""
    torch.manual_seed(0)
    B, cond_dim, A = 4, 16, 4
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    ref = _fresh_ref(policy)
    cond = torch.randn(B, cond_dim)
    chosen = torch.randn(B, A)
    rejected = torch.randn(B, A)

    loss, _ = compute_dpo_loss(policy, ref, cond, chosen, rejected, beta=0.1)
    assert loss.dim() == 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    assert all(p.grad is None for p in ref.parameters())
    print("test_dpo_loss_scalar_and_backprops_policy_only passed.")


def test_dpo_training_increases_preference_margin():
    """Optimising the DPO loss raises log π(chosen) − log π(rejected)."""
    torch.manual_seed(0)
    B, cond_dim, A = 8, 16, 4
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    ref = _fresh_ref(policy)
    cond = torch.randn(B, cond_dim)
    chosen = torch.full((B, A), 0.2)     # distinct, fixed winner/loser
    rejected = torch.full((B, A), 2.5)

    def margin():
        return (policy.log_prob(cond, chosen) - policy.log_prob(cond, rejected)).mean().item()

    before = margin()
    opt = torch.optim.Adam(policy.parameters(), lr=0.02)
    for _ in range(50):
        loss, _ = compute_dpo_loss(policy, ref, cond, chosen, rejected, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = margin()
    assert after > before, f"margin should grow: before={before:.3f} after={after:.3f}"
    print(f"test_dpo_training_increases_preference_margin passed "
          f"({before:.3f} -> {after:.3f}).")


def test_dpo_swapping_labels_flips_preference():
    """A policy trained to prefer a_w must score the swapped pair (a_l>a_w) worse.

    After training on (chosen, rejected), evaluating the loss on the *swapped*
    pair should give a higher loss than on the original — the learned preference
    is directional.
    """
    torch.manual_seed(0)
    B, cond_dim, A = 8, 16, 4
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    ref = _fresh_ref(policy)
    cond = torch.randn(B, cond_dim)
    chosen = torch.full((B, A), 0.2)
    rejected = torch.full((B, A), 2.5)

    opt = torch.optim.Adam(policy.parameters(), lr=0.02)
    for _ in range(50):
        loss, _ = compute_dpo_loss(policy, ref, cond, chosen, rejected, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        loss_correct, _ = compute_dpo_loss(policy, ref, cond, chosen, rejected, beta=0.1)
        loss_swapped, _ = compute_dpo_loss(policy, ref, cond, rejected, chosen, beta=0.1)
    assert loss_swapped > loss_correct, (
        f"swapped loss {loss_swapped:.3f} should exceed correct {loss_correct:.3f}"
    )
    print("test_dpo_swapping_labels_flips_preference passed.")


def test_sample_preference_pairs_orders_by_reward():
    """sample_preference_pairs returns a chosen action at least as good as rejected."""
    torch.manual_seed(0)
    B, cond_dim, A = 4, 16, 3
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    cond = torch.randn(B, cond_dim)
    target = torch.zeros(A)
    reward_fn = lambda a: negative_distance_reward(a, target)  # noqa: E731
    chosen, rejected = sample_preference_pairs(policy, cond, reward_fn, num_candidates=8)
    assert (reward_fn(chosen) >= reward_fn(rejected)).all()
    print("test_sample_preference_pairs_orders_by_reward passed.")


if __name__ == "__main__":
    test_build_preference_pairs_picks_best_and_worst()
    test_dpo_loss_equals_log2_when_policy_is_reference()
    test_dpo_loss_scalar_and_backprops_policy_only()
    test_dpo_training_increases_preference_margin()
    test_dpo_swapping_labels_flips_preference()
    test_sample_preference_pairs_orders_by_reward()
    print("\nAll DPO tests passed.")
