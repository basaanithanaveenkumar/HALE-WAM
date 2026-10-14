"""
Reinforcement-learning utilities for HALO-WAM (Halo-VLA).

This package fine-tunes the Halo-VLA *action policy* on top of the supervised
flow-matching pre-training that already lives in ``models.halo_vla.HaloVLM``.

Why a separate policy head?
---------------------------
The pre-trained action decoder (``models.flow_action_decoder.FlowActionDecoder``)
is a *flow-matching* model: it generates actions by integrating an ODE from
Gaussian noise.  That produces excellent samples but does **not** give a
tractable ``log π(a | s)`` — and policy-gradient and preference methods need
one (importance ratios, log-probability ratios).  A lightweight
**diagonal-Gaussian actor head** (:class:`rl.policy.GaussianActionPolicy`)
reuses the transformer's action-conditioning vector and has closed-form
``log_prob``, ``entropy`` and reparameterised sampling.

Algorithms
----------
* **REINFORCE with a learned value baseline** — :mod:`rl.reinforce`,
  :mod:`rl.value_baseline` (``scripts/train_reinforce.py``).
* **GRPO** (Group Relative Policy Optimization): critic-free PPO variant with a
  group-normalised reward baseline — :mod:`rl.grpo` (``scripts/train_grpo.py``).
* **DPO** (Direct Preference Optimization) on action chunks against a frozen
  reference policy — :mod:`rl.preference`, :mod:`rl.dpo` (``scripts/train_dpo.py``).
* **Dreamer-style latent imagination**: an RSSM world model plus an
  actor-critic trained on imagined latent rollouts — :mod:`rl.rssm`,
  :mod:`rl.actor_critic`, :mod:`rl.dreamer` (``scripts/train_dreamer.py``).

Shared: :mod:`rl.policy` (Gaussian actor head) and :mod:`rl.rewards`
(pluggable reward functions, analytic and world-model based).
"""

from rl.dpo import compute_dpo_loss
from rl.grpo import compute_grpo_loss, group_relative_advantage
from rl.policy import GaussianActionPolicy
from rl.preference import build_preference_pairs, sample_preference_pairs
from rl.rewards import (
    RewardFunction,
    WorldModelImaginationReward,
    negative_distance_reward,
)

__all__ = [
    # shared
    "GaussianActionPolicy",
    "RewardFunction",
    "negative_distance_reward",
    "WorldModelImaginationReward",
    # GRPO
    "compute_grpo_loss",
    "group_relative_advantage",
    # DPO
    "build_preference_pairs",
    "sample_preference_pairs",
    "compute_dpo_loss",
]
