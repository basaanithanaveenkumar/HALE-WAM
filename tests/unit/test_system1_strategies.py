"""System 1 policy strategies: regression, diffusion and flow matching behind one interface."""

from __future__ import annotations

import pytest
import torch

from dual_system import SYSTEM1_POLICIES, DualSystemConfig
from dual_system.system1 import ActionPolicyStrategy, DiffusionStrategy, System1Policy

CHUNK, ACTION_DIM, STATE_DIM, IMG = 4, 3, 5, 32


def make_policy(policy: str, **kw) -> System1Policy:
    cfg = DualSystemConfig(policy=policy, system1_dim=16, system1_layers=1, system1_heads=4, sample_steps=3,
                           diffusion_train_steps=20, **kw)
    torch.manual_seed(0)
    return System1Policy(cfg, IMG, STATE_DIM, CHUNK, ACTION_DIM)


def conditioned(policy: System1Policy, batch: int = 2):
    latent = torch.randn(batch, 1, 16)
    cond, mask = policy.condition(latent, torch.ones(batch, 1, dtype=torch.long), torch.rand(batch, 3, IMG, IMG),
                                  torch.randn(batch, STATE_DIM))
    return cond, mask


@pytest.mark.parametrize("name", ["regression", "diffusion", "flow_matching"])
def test_strategy_loss_is_finite_and_trains_the_shared_head(name: str) -> None:
    policy = make_policy(name)
    assert isinstance(policy.strategy, ActionPolicyStrategy) and isinstance(policy.strategy, SYSTEM1_POLICIES.get(name))
    cond, mask = conditioned(policy)
    actions = torch.randn(2, CHUNK, ACTION_DIM)
    loss = policy.loss(cond, mask, actions, torch.ones(2, CHUNK))
    loss.backward()
    assert torch.isfinite(loss) and policy.head.out[-1].weight.grad is not None


@pytest.mark.parametrize("name", ["regression", "diffusion", "flow_matching"])
def test_sampling_returns_a_chunk_and_is_seeded(name: str) -> None:
    policy = make_policy(name)
    cond, mask = conditioned(policy)
    torch.manual_seed(1)
    first = policy.sample(cond, mask)
    torch.manual_seed(1)
    second = policy.sample(cond, mask)
    assert first.shape == (2, CHUNK, ACTION_DIM) and torch.equal(first, second)


@pytest.mark.parametrize("name", ["diffusion", "flow_matching"])
def test_sample_equals_sample_from_noise_with_the_same_noise(name: str) -> None:
    """Deployment unrolls ``sample_from_noise``; training-time ``sample`` must be the same sampler."""
    policy = make_policy(name)
    cond, mask = conditioned(policy)
    torch.manual_seed(2)
    sampled = policy.sample(cond, mask, steps=3)
    torch.manual_seed(2)
    noise = torch.randn(2, CHUNK, ACTION_DIM)
    unrolled = policy.strategy.sample_from_noise(policy.head, cond, mask, noise, 3)
    assert torch.allclose(sampled, unrolled)


def test_regression_needs_no_noise_and_has_no_sampler_step() -> None:
    strategy = SYSTEM1_POLICIES.get("regression")
    assert not strategy.needs_noise and not strategy.has_sampler_step
    assert SYSTEM1_POLICIES.get("diffusion").needs_noise and SYSTEM1_POLICIES.get("flow_matching").has_sampler_step


def test_diffusion_schedule_is_a_non_persistent_buffer_with_evenly_spaced_levels() -> None:
    policy = make_policy("diffusion")
    strategy = policy.strategy
    assert isinstance(strategy, DiffusionStrategy) and policy.alpha_bar is strategy.alpha_bar
    assert not any("alpha_bar" in k for k in policy.state_dict())
    levels = strategy.levels(4)
    assert levels[0] == 19 and levels[-1] == 0 and levels == sorted(levels, reverse=True)
    assert 0 < float(strategy.alpha_bar.min()) and float(strategy.alpha_bar.max()) <= 1.0


def test_only_diffusion_policies_have_an_alpha_bar() -> None:
    with pytest.raises(AttributeError, match="alpha_bar"):
        make_policy("flow_matching").alpha_bar
