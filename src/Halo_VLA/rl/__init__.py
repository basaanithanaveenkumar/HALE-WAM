"""
Reinforcement-learning utilities for HALO-WAM (Halo-VLA).

This branch implements **Dreamer-style latent imagination**: a Recurrent
State-Space Model (RSSM) world model plus an actor-critic trained entirely on
imagined latent rollouts.

Unlike the policy-gradient / preference branches, Dreamer does not reuse the
flow decoder's conditioning directly.  It learns its *own* compact latent
dynamics from observation embeddings (e.g. pooled ViT features from Halo-VLA),
and the actor/critic act in that latent space — the most principled match for a
model that already imagines futures.

Modules
-------
* :mod:`rl.rssm`         — the RSSM world model.
* :mod:`rl.actor_critic` — Actor, Critic, and the λ-return.
* :mod:`rl.dreamer`      — world-model loss, imagination rollout, AC losses.
* :mod:`rl.rewards`      — pluggable reward functions (analytic + world-model).
"""

from rl.rssm import RSSM
from rl.actor_critic import Actor, Critic, lambda_return
from rl.dreamer import world_model_loss, imagine, actor_critic_loss
from rl.rewards import (
    RewardFunction,
    negative_distance_reward,
    WorldModelImaginationReward,
)

__all__ = [
    "RSSM",
    "Actor",
    "Critic",
    "lambda_return",
    "world_model_loss",
    "imagine",
    "actor_critic_loss",
    "RewardFunction",
    "negative_distance_reward",
    "WorldModelImaginationReward",
]
