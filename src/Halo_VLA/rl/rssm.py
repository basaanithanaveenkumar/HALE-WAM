"""
Recurrent State-Space Model (RSSM) — the world model behind Dreamer.

Concept
-------
Dreamer learns a compact *latent* model of how the world evolves, then trains an
actor and critic **entirely inside that latent model's imagination** — no
environment interaction during behaviour learning.  The world model is an RSSM
(Hafner et al., PlaNet / Dreamer).

Each latent state has two parts:

* ``h`` — a **deterministic** recurrent state (a GRU hidden vector).  It carries
  information forward through time without noise, giving the model a reliable
  memory.
* ``z`` — a **stochastic** latent (here a diagonal Gaussian).  It captures the
  uncertain, sampled part of the state.

The "model state" the heads read from is the concatenation ``feature = [h, z]``.

Two transition operators
-------------------------
* **prior**     ``p(z_t | h_t)``      — what the model *predicts* the latent is,
  using only past info (the imagination transition; needs no observation).
* **posterior** ``q(z_t | h_t, o_t)`` — the latent *corrected* by the current
  observation embedding ``o_t`` (used while training on real data).

Training pulls the prior toward the posterior with a KL term, so that at
imagination time the prior alone can roll the world forward believably.

Note on the latent: we use a **Gaussian** stochastic latent (original
PlaNet/Dreamer-v1).  DreamerV2/V3 instead use 32×32 *categorical* latents with
straight-through gradients; the structure here is identical apart from that
choice, which we keep Gaussian for simplicity and easy unit testing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


def _mlp(sizes, act=nn.ELU):
    """Small helper: build an MLP from a list of layer sizes."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class RSSM(nn.Module):
    """A compact Gaussian RSSM with reward / continue / decoder heads.

    Args:
        action_dim:   dimensionality of actions fed to the dynamics.
        obs_embed_dim:dimensionality of the observation embedding ``o_t`` (e.g.
                      a pooled ViT feature from Halo-VLA).
        deter_dim:    size of the deterministic GRU state ``h``.
        stoch_dim:    size of the stochastic latent ``z``.
        hidden_dim:   width of the internal MLPs.
        min_std:      floor on the latent std for numerical stability.
    """

    def __init__(
        self,
        action_dim: int,
        obs_embed_dim: int,
        deter_dim: int = 64,
        stoch_dim: int = 32,
        hidden_dim: int = 128,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_embed_dim = obs_embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std
        self.feature_dim = deter_dim + stoch_dim

        # Mixes the previous stochastic latent and action before the GRU.
        self.pre_gru = _mlp([stoch_dim + action_dim, hidden_dim])
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Prior  p(z_t | h_t): predict latent stats from the deterministic state.
        self.prior_net = _mlp([deter_dim, hidden_dim, 2 * stoch_dim])
        # Posterior q(z_t | h_t, o_t): correct the latent with the observation.
        self.post_net = _mlp([deter_dim + obs_embed_dim, hidden_dim, 2 * stoch_dim])

        # Heads read from feature = [h, z].
        self.reward_head = _mlp([self.feature_dim, hidden_dim, 1])
        self.continue_head = _mlp([self.feature_dim, hidden_dim, 1])      # logit
        self.decoder = _mlp([self.feature_dim, hidden_dim, obs_embed_dim])

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def initial_state(self, batch_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero (h, z) initial latent state."""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    @staticmethod
    def feature(state: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Concatenate (h, z) into the model state read by the heads/actor."""
        h, z = state
        return torch.cat([h, z], dim=-1)

    def _dist(self, stats: torch.Tensor) -> Independent:
        """Turn a [.., 2*stoch] stats vector into a diagonal Gaussian."""
        mean, std = stats.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(std) + self.min_std
        return Independent(Normal(mean, std), 1)

    # ------------------------------------------------------------------ #
    # Single-step transitions
    # ------------------------------------------------------------------ #
    def img_step(self, state, action):
        """Imagination step: advance the deterministic state and sample the
        *prior* latent (no observation involved).

        Returns:
            next_state: (h_t, z_t)
            prior:      the prior distribution over z_t.
        """
        h_prev, z_prev = state
        x = self.pre_gru(torch.cat([z_prev, action], dim=-1))
        h = self.gru(x, h_prev)                       # deterministic recurrence
        prior = self._dist(self.prior_net(h))         # p(z_t | h_t)
        z = prior.rsample()                           # reparameterised sample
        return (h, z), prior

    def obs_step(self, state, action, obs_embed):
        """Filtering step on real data: advance ``h`` via the prior, then form
        the *posterior* latent corrected by the observation embedding.

        Returns:
            next_state: (h_t, z_t)  with z_t ~ posterior
            prior:      p(z_t | h_t)
            posterior:  q(z_t | h_t, o_t)
        """
        (h, _), prior = self.img_step(state, action)
        posterior = self._dist(self.post_net(torch.cat([h, obs_embed], dim=-1)))
        z = posterior.rsample()
        return (h, z), prior, posterior

    # ------------------------------------------------------------------ #
    # Sequence rollout on real data
    # ------------------------------------------------------------------ #
    def observe(self, obs_embeds: torch.Tensor, actions: torch.Tensor, state=None):
        """Run the posterior filter over a sequence of observations/actions.

        Args:
            obs_embeds: [B, T, obs_embed_dim] observation embeddings.
            actions:    [B, T, action_dim] actions taken *before* each obs.
            state:      optional initial (h, z); defaults to zeros.

        Returns:
            dict with stacked ``features`` [B, T, F], the per-step ``priors`` and
            ``posteriors`` distributions, and the final ``state``.
        """
        B, T, _ = obs_embeds.shape
        if state is None:
            state = self.initial_state(B, obs_embeds.device)

        feats, priors, posts = [], [], []
        for t in range(T):
            state, prior, post = self.obs_step(state, actions[:, t], obs_embeds[:, t])
            feats.append(self.feature(state))
            priors.append(prior)
            posts.append(post)

        return {
            "features": torch.stack(feats, dim=1),    # [B, T, F]
            "priors": priors,                          # list[T] of dists
            "posteriors": posts,                       # list[T] of dists
            "state": state,
        }
