"""ActionPolicyStrategy: how System 1 trains and samples an action chunk."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig


class ActionPolicyStrategy(nn.Module, ABC):
    """The training objective and sampler of one System 1 policy kind.

    Pattern: Strategy + Registry. :class:`~dual_system.system1.system1_policy.System1Policy`
    owns the shared conditioning and the :class:`ActionTransformer`; a strategy owns what differs
    between regression, diffusion and flow matching. It is an ``nn.Module`` only so that
    schedule buffers follow the policy across devices; it has no parameters, and the
    transformer is passed in per call, so checkpoints keep their ``head.*`` keys.

    Attributes:
        needs_noise: whether sampling starts from Gaussian noise.
        has_sampler_step: whether sampling iterates the transformer (False for one-shot regression).

    Args:
        config: dual-system settings.
    """

    needs_noise: bool = True
    has_sampler_step: bool = True

    def __init__(self, config: DualSystemConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def loss(self, head: nn.Module, cond: torch.Tensor, cond_mask: torch.Tensor, actions: torch.Tensor,
             weight: torch.Tensor) -> torch.Tensor:
        """Training loss on a target chunk.

        Args:
            head: the shared action transformer.
            cond: conditioning tokens ``(B, S, D)``.
            cond_mask: conditioning mask ``(B, S)``.
            actions: target chunk ``(B, chunk, action_dim)``.
            weight: per-step validity ``(B, chunk, 1)`` in ``actions``' dtype.

        Returns:
            A scalar loss.
        """

    @abstractmethod
    def sample_from_noise(self, head: nn.Module, cond: torch.Tensor, cond_mask: torch.Tensor,
                          noise: torch.Tensor | None, steps: int) -> torch.Tensor:
        """Deterministic sampler given the initial noise (what deployment unrolls).

        Args:
            head: the shared action transformer.
            cond: conditioning tokens ``(B, S, D)``.
            cond_mask: conditioning mask ``(B, S)``.
            noise: initial ``(B, chunk, action_dim)`` noise; ignored when ``needs_noise`` is False.
            steps: sampling steps.

        Returns:
            The action chunk ``(B, chunk, action_dim)``.
        """

    def prepare(self, steps: int) -> None:
        """Resolve anything that must be concrete before export tracing (default: nothing)."""

    @torch.no_grad()
    def sample(self, head: nn.Module, cond: torch.Tensor, cond_mask: torch.Tensor, chunk: int,
               action_dim: int, steps: int) -> torch.Tensor:
        """Draw noise (if the policy needs it) and sample an action chunk.

        Args:
            head: the shared action transformer.
            cond: conditioning tokens ``(B, S, D)``.
            cond_mask: conditioning mask ``(B, S)``.
            chunk: actions per chunk.
            action_dim: action size.
            steps: sampling steps.

        Returns:
            The action chunk ``(B, chunk, action_dim)``.
        """
        noise = torch.randn(cond.size(0), chunk, action_dim, device=cond.device) if self.needs_noise else None
        return self.sample_from_noise(head, cond, cond_mask, noise, steps)

    @staticmethod
    def masked_mean(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Sum of ``error * weight`` over valid steps divided by the number of valid values."""
        return (error * weight).sum() / (weight.sum() * error.size(-1)).clamp_min(1.0)
