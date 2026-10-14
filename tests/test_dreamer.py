"""
Unit tests for Dreamer-style latent imagination (branch ``rl/dreamer``).

Run with:
    pytest tests/test_dreamer.py -o addopts="" -q
or standalone:
    python tests/test_dreamer.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from rl.rssm import RSSM
from rl.actor_critic import Actor, Critic, lambda_return
from rl.dreamer import world_model_loss, imagine, actor_critic_loss


def _make(action_dim=3, obs_dim=8):
    rssm = RSSM(action_dim=action_dim, obs_embed_dim=obs_dim,
                deter_dim=16, stoch_dim=8, hidden_dim=32)
    actor = Actor(feature_dim=rssm.feature_dim, action_dim=action_dim, hidden_dim=32)
    critic = Critic(feature_dim=rssm.feature_dim, hidden_dim=32)
    return rssm, actor, critic


def test_lambda_return_matches_hand_computation():
    """Verify the λ-return against a closed-form hand calculation.

    gamma=0.9, lam=0.5, continue=1, value=[5,3] (bootstrap=v[-1]=3), reward=[1,2]:
        G_1 = r1 + γ·v1            = 2 + 0.9·3        = 4.7
        G_0 = r0 + γ(1-λ)v1 + γλG_1 = 1 + 0.45·3 + 0.45·4.7 = 4.465
    """
    reward = torch.tensor([[1.0, 2.0]])
    value = torch.tensor([[5.0, 3.0]])
    cont = torch.ones(1, 2)
    out = lambda_return(reward, value, cont, gamma=0.9, lam=0.5)
    expected = torch.tensor([[4.465, 4.7]])
    assert torch.allclose(out, expected, atol=1e-5), f"{out} != {expected}"

    # λ=0 must reduce to the one-step TD target r + γ·c·V(s_{t+1}).
    out0 = lambda_return(reward, value, cont, gamma=0.9, lam=0.0)
    td = torch.tensor([[1.0 + 0.9 * 3.0, 2.0 + 0.9 * 3.0]])
    assert torch.allclose(out0, td, atol=1e-5), f"{out0} != {td}"
    print("test_lambda_return_matches_hand_computation passed.")


def test_rssm_observe_and_imagine_shapes():
    torch.manual_seed(0)
    rssm, actor, _ = _make(action_dim=3, obs_dim=8)
    B, T = 4, 5
    obs = torch.randn(B, T, 8)
    acts = torch.randn(B, T, 3)

    out = rssm.observe(obs, acts)
    assert out["features"].shape == (B, T, rssm.feature_dim)
    assert len(out["priors"]) == T and len(out["posteriors"]) == T

    # imagine from the final posterior state.
    traj = imagine(rssm, actor, out["state"], horizon=7)
    assert traj["features"].shape == (B, 7, rssm.feature_dim)
    assert traj["actions"].shape == (B, 7, 3)
    assert traj["rewards"].shape == (B, 7)
    assert traj["continues"].shape == (B, 7)
    assert (traj["continues"] >= 0).all() and (traj["continues"] <= 1).all()
    print("test_rssm_observe_and_imagine_shapes passed.")


def test_world_model_loss_decreases():
    """The RSSM should overfit a small fixed batch (loss goes down)."""
    torch.manual_seed(0)
    rssm, _, _ = _make(action_dim=3, obs_dim=8)
    B, T = 4, 6
    obs = torch.randn(B, T, 8)
    acts = torch.randn(B, T, 3)
    rewards = torch.randn(B, T)

    opt = torch.optim.Adam(rssm.parameters(), lr=3e-3)
    first = last = None
    for i in range(80):
        loss, metrics, _ = world_model_loss(rssm, obs, acts, rewards, free_nats=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = metrics["wm_loss"]
        last = metrics["wm_loss"]
    assert last < first, f"world-model loss should decrease: {first} -> {last}"
    print(f"test_world_model_loss_decreases passed ({first:.3f} -> {last:.3f}).")


def test_actor_critic_losses_scalar_and_isolated():
    """Actor/critic losses are scalar, backprop into actor & critic, and do NOT
    leak gradients into the (fixed) world model."""
    torch.manual_seed(0)
    rssm, actor, critic = _make(action_dim=3, obs_dim=8)
    B, T = 4, 5
    obs = torch.randn(B, T, 8)
    acts = torch.randn(B, T, 3)

    # Seed imagination from posterior states of real data.
    with torch.no_grad():
        roll = rssm.observe(obs, acts)
    init_state = roll["state"]

    actor_loss, critic_loss, metrics = actor_critic_loss(
        rssm, actor, critic, init_state, horizon=8
    )
    assert actor_loss.dim() == 0 and critic_loss.dim() == 0

    (actor_loss + critic_loss).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    # World model must stay fixed during behaviour learning.
    assert all(p.grad is None for p in rssm.parameters()), "world model got gradients!"
    print("test_actor_critic_losses_scalar_and_isolated passed.")


def test_actor_critic_improves_imagined_return():
    """Optimising the actor raises the imagined return under a shaped reward.

    We give the world model a reward head that is already informative by training
    it briefly to predict a reward that favours one latent direction, then check
    the actor learns to collect more of it in imagination.
    """
    torch.manual_seed(0)
    rssm, actor, critic = _make(action_dim=3, obs_dim=8)
    B, T = 6, 5
    obs = torch.randn(B, T, 8)
    acts = torch.randn(B, T, 3)
    # Reward proportional to the first obs feature so dynamics carry real signal.
    rewards = obs[..., 0]

    wm_opt = torch.optim.Adam(rssm.parameters(), lr=3e-3)
    for _ in range(120):
        loss, _, _ = world_model_loss(rssm, obs, acts, rewards, free_nats=0.0)
        wm_opt.zero_grad()
        loss.backward()
        wm_opt.step()

    with torch.no_grad():
        init_state = rssm.observe(obs, acts)["state"]

    def avg_return():
        with torch.no_grad():
            return imagine(rssm, actor, init_state, horizon=10)["rewards"].sum(1).mean().item()

    before = avg_return()
    ac_opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-3)
    for _ in range(150):
        a_loss, c_loss, _ = actor_critic_loss(rssm, actor, critic, init_state, horizon=10)
        ac_opt.zero_grad()
        (a_loss + c_loss).backward()
        ac_opt.step()
    after = avg_return()

    assert after > before, f"imagined return should improve: {before:.3f} -> {after:.3f}"
    print(f"test_actor_critic_improves_imagined_return passed ({before:.3f} -> {after:.3f}).")


if __name__ == "__main__":
    test_lambda_return_matches_hand_computation()
    test_rssm_observe_and_imagine_shapes()
    test_world_model_loss_decreases()
    test_actor_critic_losses_scalar_and_isolated()
    test_actor_critic_improves_imagined_return()
    print("\nAll Dreamer tests passed.")
