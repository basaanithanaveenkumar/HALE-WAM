"""
Reinforcement-learning utilities for HALO-WAM (Halo-VLA).

This package fine-tunes the Halo-VLA *action policy* with reinforcement learning
on top of the supervised flow-matching pre-training that already lives in
``models.halo_vla.HaloVLM``.

Why a separate policy head?
---------------------------
The pre-trained action decoder (``models.flow_action_decoder.FlowActionDecoder``)
is a *flow-matching* model: it generates actions by integrating an ODE from
Gaussian noise.  That produces excellent samples but does **not** give a
tractable ``log π(a | s)`` — and almost every policy-gradient / preference RL
method needs one.  We therefore add a lightweight **diagonal-Gaussian actor
head** (:class:`rl.policy.GaussianActionPolicy`) that reuses the transformer's
action-conditioning vector.  The Gaussian head has closed-form ``log_prob``,
``entropy`` and reparameterised sampling, so it slots directly into REINFORCE,
GRPO and DPO.

This branch implements **REINFORCE with a learned value baseline**.
"""

from rl.policy import GaussianActionPolicy
from rl.rewards import (
    RewardFunction,
    negative_distance_reward,
    WorldModelImaginationReward,
)
from rl.value_baseline import ValueBaseline
from rl.reinforce import compute_reinforce_loss

__all__ = [
    "GaussianActionPolicy",
    "RewardFunction",
    "negative_distance_reward",
    "WorldModelImaginationReward",
    "ValueBaseline",
    "compute_reinforce_loss",
]
