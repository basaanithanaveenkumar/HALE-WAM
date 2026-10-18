"""Dual-system System 1 in one call: condition tokens + unrolled sampler (noise as input)."""

from __future__ import annotations

import torch
from torch import nn


class System1PolicyGraph(nn.Module):
    """``System1Policy.condition`` + ``System1Policy.sample`` with deterministic noise.

    The sampler is the policy strategy's own ``sample_from_noise`` (one implementation for
    training-time sampling and export). Schedules are resolved at export time: flow matching
    unrolls ``steps`` Euler steps (timestep ``t * 1000`` like training); diffusion unrolls DDIM
    over the evenly spaced noise levels, whose ``alpha_bar`` values become constants; regression
    is a single head call and takes no noise input.

    Args:
        dual: the built ``DualSystemVLA``.
        steps: sampling steps (default: ``config.sample_steps``).
    """

    def __init__(self, dual: nn.Module, steps: int | None = None) -> None:
        super().__init__()
        self.system1 = dual.system1
        self.policy = dual.config.policy
        self.steps = int(steps or dual.config.sample_steps)
        self.system1.strategy.prepare(self.steps)  # concrete noise levels before tracing

    def forward(
        self,
        latent: torch.Tensor,
        latent_mask: torch.Tensor,
        image: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Condition on the latent and run the policy's sampler from ``noise``.

        Args:
            latent: System 2 latent ``(B, K, system1_dim)``.
            latent_mask: its mask ``(B, K)``.
            image: camera frame ``(B, 3, H, W)``.
            state: proprioception ``(B, state_dim)``.
            noise: initial noise ``(B, chunk, action_dim)``; ``None`` for regression.

        Returns:
            The action chunk ``(B, chunk, action_dim)``.
        """
        cond, cond_mask = self.system1.condition(latent, latent_mask, image, state)
        return self.system1.strategy.sample_from_noise(
            self.system1.head, cond, cond_mask, noise, self.steps
        )
