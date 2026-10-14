"""
Unit tests for REINFORCE-with-baseline (branch ``rl/reinforce-baseline``).

Run with:
    pytest tests/test_reinforce.py -p no:cacheprovider -q
or standalone:
    python tests/test_reinforce.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from rl.policy import GaussianActionPolicy
from rl.value_baseline import ValueBaseline
from rl.reinforce import compute_reinforce_loss
from rl.rewards import negative_distance_reward


def test_policy_shapes_and_gradients():
    """Policy sampling/scoring returns correctly shaped, differentiable tensors."""
    torch.manual_seed(0)
    B, cond_dim, A = 4, 16, 8
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    cond = torch.randn(B, cond_dim)

    action, log_prob = policy.sample(cond)
    assert action.shape == (B, A)
    assert log_prob.shape == (B,)
    # sampled action must be detached (score-function estimator), log_prob must
    # carry gradient through the distribution parameters.
    assert not action.requires_grad
    assert log_prob.requires_grad

    # entropy and explicit log_prob also have the right shape.
    assert policy.entropy(cond).shape == (B,)
    assert policy.log_prob(cond, action).shape == (B,)
    print("test_policy_shapes_and_gradients passed.")


def test_reinforce_loss_is_scalar_and_backprops():
    """The combined loss is a scalar and produces gradients for policy + critic."""
    torch.manual_seed(0)
    B, cond_dim, A = 4, 16, 8
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    value = ValueBaseline(cond_dim, hidden_dim=32)
    cond = torch.randn(B, cond_dim)
    target = torch.randn(A)

    loss, metrics = compute_reinforce_loss(
        policy, value, cond, lambda a: negative_distance_reward(a, target)
    )
    assert loss.dim() == 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in value.parameters())
    assert set(["policy_loss", "value_loss", "entropy", "reward_mean"]).issubset(metrics)
    print("test_reinforce_loss_is_scalar_and_backprops passed.")


def test_value_baseline_learns_mean_reward():
    """The critic V(s) converges to the expected reward E_a[R(a)] of the state."""
    torch.manual_seed(0)
    B, cond_dim, A = 6, 16, 4
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    value = ValueBaseline(cond_dim, hidden_dim=32)
    cond = torch.randn(B, cond_dim)
    target = torch.randn(A)
    reward_fn = lambda a: negative_distance_reward(a, target, offset=10.0)  # noqa: E731

    vopt = torch.optim.Adam(value.parameters(), lr=1e-2)
    for _ in range(500):
        a, _ = policy.sample(cond)
        r = reward_fn(a).detach()
        vloss = F.mse_loss(value(cond), r)
        vopt.zero_grad()
        vloss.backward()
        vopt.step()

    # Monte-Carlo estimate of the true E_a[R] per state.
    with torch.no_grad():
        mc = torch.stack([reward_fn(policy.sample(cond)[0]) for _ in range(400)]).mean(0)
        v = value(cond)
    assert torch.allclose(v, mc, atol=0.5), f"V={v} vs MC={mc}"
    print("test_value_baseline_learns_mean_reward passed.")


def test_baseline_reduces_gradient_variance():
    """A fitted baseline lowers the variance of the policy-gradient estimator.

    With a large reward offset, the raw-reward estimator is dominated by the
    constant offset times ∇logπ (high variance); subtracting V(s) ≈ E[R] removes
    that offset, so the with-baseline estimator must have lower variance.
    """
    torch.manual_seed(0)
    B, cond_dim, A = 8, 16, 4
    policy = GaussianActionPolicy(cond_dim, A, hidden_dim=32)
    value = ValueBaseline(cond_dim, hidden_dim=32)
    cond = torch.randn(B, cond_dim)
    target = torch.randn(A)
    reward_fn = lambda a: negative_distance_reward(a, target, offset=100.0)  # noqa: E731

    # Fit the baseline to ~E[R] so it can actually cancel the offset.
    vopt = torch.optim.Adam(value.parameters(), lr=1e-2)
    for _ in range(400):
        a, _ = policy.sample(cond)
        r = reward_fn(a).detach()
        vloss = F.mse_loss(value(cond), r)
        vopt.zero_grad()
        vloss.backward()
        vopt.step()

    def policy_grad(use_baseline):
        policy.zero_grad()
        value.zero_grad()
        loss, _ = compute_reinforce_loss(
            policy, value, cond, reward_fn,
            use_baseline=use_baseline, entropy_coef=0.0, value_coef=0.0,
        )
        loss.backward()
        return torch.cat([p.grad.flatten() for p in policy.parameters()])

    def grad_variance(use_baseline, trials=200):
        grads = torch.stack([policy_grad(use_baseline) for _ in range(trials)])
        return grads.var(dim=0, unbiased=False).mean().item()

    var_with = grad_variance(True)
    var_without = grad_variance(False)
    assert var_with < var_without, f"with={var_with} should be < without={var_without}"
    print(f"test_baseline_reduces_gradient_variance passed "
          f"(with={var_with:.3g} < without={var_without:.3g}).")


def test_advantage_sign_matches_reward_ordering():
    """Closer-to-target actions get higher reward and positive advantage."""
    torch.manual_seed(0)
    A = 4
    target = torch.zeros(A)
    good = torch.full((1, A), 0.1)   # near target -> high reward
    bad = torch.full((1, A), 5.0)    # far  target -> low  reward
    r_good = negative_distance_reward(good, target)
    r_bad = negative_distance_reward(bad, target)
    assert r_good.item() > r_bad.item()

    # With baseline = mean of the two, the good action has positive advantage.
    baseline = (r_good + r_bad) / 2
    assert (r_good - baseline).item() > 0
    assert (r_bad - baseline).item() < 0
    print("test_advantage_sign_matches_reward_ordering passed.")


if __name__ == "__main__":
    test_policy_shapes_and_gradients()
    test_reinforce_loss_is_scalar_and_backprops()
    test_value_baseline_learns_mean_reward()
    test_baseline_reduces_gradient_variance()
    test_advantage_sign_matches_reward_ordering()
    print("\nAll REINFORCE tests passed.")
