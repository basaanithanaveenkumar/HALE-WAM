"""Reward functions for RL fine-tuning of Halo-VLA.

A reward ``R(a, s)`` is the scalar feedback the agent tries to maximise.  In a
world-model setting the reward typically measures *how good an imagined outcome
is* — e.g. how close the predicted future is to a goal.

This module provides:

* :class:`RewardFunction` — a tiny structural-typing protocol so any callable
  ``actions -> [B]`` can be used as a reward (dependency inversion: the RL
  losses depend on this abstraction, not on a concrete reward).
* :func:`negative_distance_reward` — a deterministic analytic reward used by the
  unit tests (reproducible, no model required).
* :class:`WorldModelImaginationReward` — scores Halo-VLA's *imagined* future
  frames against a goal frame, i.e. reward straight from the world model."""

from .common import negative_distance_reward  # noqa: F401
from .reward_function import RewardFunction  # noqa: F401
from .world_model_imagination_reward import WorldModelImaginationReward  # noqa: F401

__all__ = [
    "RewardFunction",
    "WorldModelImaginationReward",
    "negative_distance_reward",
]
