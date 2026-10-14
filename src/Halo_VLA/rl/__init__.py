"""
Reinforcement-learning utilities for HALO-WAM (Halo-VLA).

This package fine-tunes the Halo-VLA *action policy* with preference learning on
top of the supervised flow-matching pre-training that already lives in
``models.halo_vla.HaloVLM``.

Why a separate policy head?
---------------------------
The pre-trained action decoder (``models.flow_action_decoder.FlowActionDecoder``)
is a *flow-matching* model: it generates actions by integrating an ODE from
Gaussian noise.  That produces excellent samples but does **not** give a
tractable ``log π(a | s)`` — and DPO is built entirely out of log-probability
ratios.  We therefore add a lightweight **diagonal-Gaussian actor head**
(:class:`rl.policy.GaussianActionPolicy`) that reuses the transformer's
action-conditioning vector and has a closed-form log-prob.

This branch implements **Direct Preference Optimization (DPO)** on action chunks.
"""

from rl.policy import GaussianActionPolicy
from rl.rewards import (
    RewardFunction,
    negative_distance_reward,
    WorldModelImaginationReward,
)
from rl.preference import build_preference_pairs, sample_preference_pairs
from rl.dpo import compute_dpo_loss

__all__ = [
    "GaussianActionPolicy",
    "RewardFunction",
    "negative_distance_reward",
    "WorldModelImaginationReward",
    "build_preference_pairs",
    "sample_preference_pairs",
    "compute_dpo_loss",
]
