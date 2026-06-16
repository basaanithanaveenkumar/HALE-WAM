"""
Reward functions for RL fine-tuning of Halo-VLA.

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
  frames against a goal frame, i.e. reward straight from the world model.
"""

from __future__ import annotations

from typing import Protocol

import torch


class RewardFunction(Protocol):
    """Anything callable that maps a batch of actions to a reward per sample.

    Implementations return a tensor of shape ``[B]`` (one scalar reward per
    sample).  Rewards do **not** need gradients — RL losses detach them.
    """

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:  # pragma: no cover - protocol
        ...


def negative_distance_reward(
    actions: torch.Tensor,
    target: torch.Tensor,
    squared: bool = True,
    offset: float = 0.0,
) -> torch.Tensor:
    """Reward = (offset) − distance(actions, target).

    The closer the action is to ``target``, the higher (less negative) the
    reward — a clean, convex, fully-deterministic signal that makes the unit
    tests reproducible.

    Args:
        actions: [B, A] sampled actions.
        target:  [A] or [B, A] goal action(s).
        squared: if True use squared-L2 (smooth); else L2 norm.
        offset:  constant added to every reward.

    Returns:
        reward: [B].
    """
    if target.dim() == 1:
        target = target.unsqueeze(0)            # [1, A] broadcasts over batch
    diff = actions - target                     # [B, A]
    dist_sq = diff.pow(2).sum(dim=-1)           # [B]
    dist = dist_sq if squared else dist_sq.clamp_min(1e-12).sqrt()
    return offset - dist


class WorldModelImaginationReward:
    """Reward from imagining the future with Halo-VLA's DiT world model.

    The reward is the negative pixel distance between the *last imagined future
    frame* and a goal frame::

        R = − mean_pixels( (imagined_future − goal_frame)² )

    Conceptually: the policy/observation context flows through the transformer
    and conditions the DiT video predictor, which "dreams" what happens next;
    we reward dreams that look like the goal.  This keeps reward grounded in the
    learned world model rather than in privileged simulator state.

    Note: this is the *illustrative* reward for real training.  The unit tests
    use :func:`negative_distance_reward` instead, because it is deterministic
    and needs no model.
    """

    def __init__(
        self,
        model,
        goal_frame: torch.Tensor,
        num_frames: int = 1,
        num_ode_steps: int = 20,
    ):
        """
        Args:
            model:      a ``models.halo_vla.HaloVLM`` with a visual predictor.
            goal_frame: [3, H, W] or [B, 3, H, W] target appearance.
            num_frames: how many future frames to imagine (reward uses the last).
            num_ode_steps: DiT Euler-integration steps per imagined frame.
        """
        self.model = model
        self.goal_frame = goal_frame
        self.num_frames = num_frames
        self.num_ode_steps = num_ode_steps

    @torch.no_grad()
    def __call__(
        self,
        past_rgb: torch.Tensor,
        visual_context_emb: torch.Tensor | None = None,
        world_video_query_hiddens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Imagine the future from ``past_rgb`` and score it against the goal.

        Args:
            past_rgb: [B, T, 3, H, W] or [B, 3, H, W] observed context frames.
        Returns:
            reward: [B].
        """
        preds = self.model.predict_visual_future(
            past_rgb=past_rgb,
            visual_context_emb=visual_context_emb,
            num_frames=self.num_frames,
            world_video_query_hiddens=world_video_query_hiddens,
            num_ode_steps=self.num_ode_steps,
        )
        if not preds or "rgb" not in preds:
            raise RuntimeError("World model produced no RGB prediction.")
        imagined = preds["rgb"][:, -1]                       # [B, 3, H, W]

        goal = self.goal_frame
        if goal.dim() == 3:
            goal = goal.unsqueeze(0)                         # [1, 3, H, W]
        goal = goal.to(imagined.device, imagined.dtype)
        return -((imagined - goal) ** 2).mean(dim=(1, 2, 3))  # [B]
