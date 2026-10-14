"""
Dreamer training objectives: world-model learning + actor-critic in imagination.

Dreamer alternates two phases:

1. **World-model learning** (:func:`world_model_loss`) — fit the RSSM to real
   sequences so it can reconstruct observations, predict rewards/continues, and
   so its prior matches its posterior (the KL term).  This is the only phase that
   touches real data.

2. **Behaviour learning in imagination** (:func:`actor_critic_loss`) — starting
   from the posterior states of real data, roll the RSSM *prior* forward under
   the actor for a short horizon, entirely in latent space.  The critic regresses
   onto λ-returns of the imagined rewards; the actor is updated with REINFORCE on
   those returns plus an entropy bonus.  No environment steps, no pixels.

Gradient flow note
------------------
During behaviour learning we **detach** the imagined latent features from the
world model: the actor/critic optimiser must not silently update the world model.
The actor uses the score-function (REINFORCE) estimator — ``log π(a)`` recomputed
with gradients on fixed sampled actions, weighted by a detached advantage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from rl.rssm import RSSM
from rl.actor_critic import Actor, Critic, lambda_return


# ---------------------------------------------------------------------------
# Phase 1 — world model
# ---------------------------------------------------------------------------
def world_model_loss(
    rssm: RSSM,
    obs_embeds: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    continues: torch.Tensor | None = None,
    *,
    free_nats: float = 1.0,
    kl_scale: float = 1.0,
    kl_balance: float = 0.8,
) -> tuple[torch.Tensor, dict, dict]:
    """Fit the RSSM to a batch of real sequences.

    Args:
        rssm:       the world model.
        obs_embeds: [B, T, obs_dim] observation embeddings.
        actions:    [B, T, action_dim].
        rewards:    [B, T] observed/assigned rewards.
        continues:  [B, T] continuation flags ∈ {0,1}; defaults to all ones.
        free_nats:  KL below this many nats is not penalised ("free bits"), which
                    stops the KL from crushing the latent early in training.
        kl_scale:   weight on the KL term.
        kl_balance: KL-balancing coefficient α (DreamerV2).  Blends
                    ``KL(stopgrad(post) ‖ prior)`` (trains the prior toward the
                    posterior, weight α) and ``KL(post ‖ stopgrad(prior))``
                    (regularises the posterior, weight 1−α).

    Returns:
        (loss, metrics, rollout) where ``rollout`` holds the posterior
        ``features`` used to seed imagination.
    """
    B, T, _ = obs_embeds.shape
    if continues is None:
        continues = torch.ones(B, T, device=obs_embeds.device)

    rollout = rssm.observe(obs_embeds, actions)
    feats = rollout["features"]                          # [B, T, F]

    # Reconstruction: decode the observation embedding back from the feature.
    recon = rssm.decoder(feats)
    recon_loss = F.mse_loss(recon, obs_embeds)

    # Reward prediction.
    reward_pred = rssm.reward_head(feats).squeeze(-1)    # [B, T]
    reward_loss = F.mse_loss(reward_pred, rewards)

    # Continue prediction (Bernoulli via BCE-with-logits).
    cont_logit = rssm.continue_head(feats).squeeze(-1)   # [B, T]
    continue_loss = F.binary_cross_entropy_with_logits(cont_logit, continues)

    # KL between posterior and prior at every step, with KL balancing + free nats.
    kl_values = []
    for prior, post in zip(rollout["priors"], rollout["posteriors"]):
        post_sg = _detach_dist(post)
        prior_sg = _detach_dist(prior)
        kl_lhs = kl_divergence(post_sg, prior).mean()    # train prior -> posterior
        kl_rhs = kl_divergence(post, prior_sg).mean()    # regularise posterior
        kl_values.append(kl_balance * kl_lhs + (1.0 - kl_balance) * kl_rhs)
    kl = torch.stack(kl_values).mean()
    kl_clipped = torch.clamp(kl, min=free_nats)          # free-nats floor

    loss = recon_loss + reward_loss + continue_loss + kl_scale * kl_clipped

    metrics = {
        "wm_loss": float(loss.detach()),
        "recon_loss": float(recon_loss.detach()),
        "reward_loss": float(reward_loss.detach()),
        "continue_loss": float(continue_loss.detach()),
        "kl": float(kl.detach()),
    }
    return loss, metrics, rollout


def _detach_dist(dist):
    """Return a copy of an Independent(Normal) with detached parameters
    (stop-gradient), used for KL balancing."""
    base = dist.base_dist
    from torch.distributions import Independent, Normal
    return Independent(Normal(base.loc.detach(), base.scale.detach()), 1)


# ---------------------------------------------------------------------------
# Phase 2 — imagination + actor-critic
# ---------------------------------------------------------------------------
@torch.no_grad()
def imagine(rssm: RSSM, actor: Actor, init_state, horizon: int):
    """Roll the RSSM prior forward under the actor, purely in latent space.

    The dynamics rollout runs under ``no_grad`` — gradients for the actor are
    obtained later by recomputing ``log π`` on the recorded actions (REINFORCE),
    and the critic trains on detached features.  This keeps the world model
    fixed during behaviour learning.

    Args:
        rssm:       the (fixed) world model.
        actor:      the policy used to choose imagined actions.
        init_state: (h, z) seed state(s), shape [M, ·] each.
        horizon:    number of imagination steps H.

    Returns:
        dict with ``features`` [M, H, F], ``actions`` [M, H, A],
        ``rewards`` [M, H], ``continues`` [M, H].
    """
    state = init_state
    feats, acts, rews, conts = [], [], [], []
    for _ in range(horizon):
        feat = rssm.feature(state)
        action, _ = actor.sample(feat)
        state, _ = rssm.img_step(state, action)
        next_feat = rssm.feature(state)
        feats.append(next_feat)
        acts.append(action)
        rews.append(rssm.reward_head(next_feat).squeeze(-1))
        conts.append(torch.sigmoid(rssm.continue_head(next_feat).squeeze(-1)))
    return {
        "features": torch.stack(feats, dim=1),     # [M, H, F]
        "actions": torch.stack(acts, dim=1),       # [M, H, A]
        "rewards": torch.stack(rews, dim=1),       # [M, H]
        "continues": torch.stack(conts, dim=1),    # [M, H]
    }


def actor_critic_loss(
    rssm: RSSM,
    actor: Actor,
    critic: Critic,
    init_state,
    *,
    horizon: int = 10,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute actor and critic losses from imagined trajectories.

    Args:
        rssm/actor/critic: the model and behaviour networks.
        init_state:        (h, z) seed states (e.g. flattened posterior states).
        horizon:           imagination horizon H.
        gamma, lam:        discount and λ for the λ-return.
        entropy_coef:      weight on the actor entropy bonus.

    Returns:
        (actor_loss, critic_loss, metrics).
    """
    traj = imagine(rssm, actor, init_state, horizon)
    feats = traj["features"]                       # [M, H, F] (detached: no_grad)
    actions = traj["actions"]                      # [M, H, A]
    rewards = traj["rewards"]                      # [M, H]
    continues = traj["continues"]                  # [M, H]

    # Critic values of the imagined states (with grad for the critic loss).
    values = critic(feats)                         # [M, H]

    # λ-return targets — computed from detached values, then detached entirely so
    # neither actor nor critic backprops through the target ("semi-gradient").
    returns = lambda_return(rewards, values.detach(), continues, gamma, lam).detach()

    # Critic: regress V(s) onto the λ-return.
    critic_loss = F.mse_loss(values, returns)

    # Actor: REINFORCE on the advantage (returns − value), plus entropy bonus.
    M, H, F_dim = feats.shape
    flat_feat = feats.reshape(M * H, F_dim)
    flat_act = actions.reshape(M * H, -1)
    log_prob = actor.log_prob(flat_feat, flat_act).reshape(M, H)
    advantage = (returns - values.detach())
    actor_loss = -(log_prob * advantage).mean() - entropy_coef * actor.entropy(flat_feat).mean()

    metrics = {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "imagined_return": float(returns.mean()),
        "imagined_reward": float(rewards.mean()),
        "value_mean": float(values.mean().detach()),
    }
    return actor_loss, critic_loss, metrics
