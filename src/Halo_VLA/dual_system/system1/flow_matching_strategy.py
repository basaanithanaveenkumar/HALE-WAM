"""FlowMatchingStrategy: rectified flow with Euler sampling (pi0)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .action_policy_strategy import ActionPolicyStrategy
from .system1_registry import SYSTEM1_POLICIES

#: The head sees ``t * TIME_SCALE`` so flow and diffusion share one timestep embedding range.
TIME_SCALE = 1000.0


@SYSTEM1_POLICIES.register_module(name="flow_matching")
class FlowMatchingStrategy(ActionPolicyStrategy):
    """Rectified flow ``x_t = (1 - t) x_0 + t x_1`` with velocity ``x_1 - x_0``.

    (Lipman et al., 2023; Liu et al., 2023.) Sampling is ``steps`` Euler steps: O(steps) head
    evaluations.
    """

    def loss(
        self,
        head: nn.Module,
        cond: torch.Tensor,
        cond_mask: torch.Tensor,
        actions: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared error of the predicted velocity."""
        b = actions.size(0)
        noise = torch.randn_like(actions)
        t = torch.rand(b, device=actions.device)
        noisy = (1 - t.view(b, 1, 1)) * noise + t.view(b, 1, 1) * actions
        pred = head(cond, cond_mask, noisy, t * TIME_SCALE)
        return self.masked_mean((pred - (actions - noise)) ** 2, weight)

    def sample_from_noise(
        self,
        head: nn.Module,
        cond: torch.Tensor,
        cond_mask: torch.Tensor,
        noise: torch.Tensor | None,
        steps: int,
    ) -> torch.Tensor:
        """Euler integration of the learned velocity field from ``t = 0`` to ``1``."""
        x = noise
        for i in range(steps):
            t = torch.full_like(x[:, 0, 0], i / steps)
            x = x + head(cond, cond_mask, x, t * TIME_SCALE) / steps
        return x
