"""Dual-system System 1 in one call: condition tokens + unrolled sampler (noise as input)."""

from __future__ import annotations

import torch
from torch import nn


class System1PolicyGraph(nn.Module):
    """``System1Policy.condition`` + ``System1Policy.sample`` with deterministic noise.

    Sampling schedules are resolved at export time: flow matching unrolls ``steps`` Euler steps
    (timestep ``t * 1000`` like training); diffusion unrolls DDIM over the evenly spaced noise
    levels, whose ``alpha_bar`` values become constants; regression is a single head call and
    takes no noise input.

    Args:
        dual: the built ``DualSystemVLA``.
        steps: sampling steps (default: ``config.sample_steps``).
    """

    def __init__(self, dual: nn.Module, steps: int | None = None) -> None:
        super().__init__()
        self.system1 = dual.system1
        self.policy = dual.config.policy
        self.steps = int(steps or dual.config.sample_steps)
        if self.policy == "diffusion":
            n = self.system1.alpha_bar.numel()
            self.levels = torch.linspace(n - 1, 0, self.steps).round().long().tolist()

    def forward(self, latent: torch.Tensor, latent_mask: torch.Tensor, image: torch.Tensor, state: torch.Tensor,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        cond, cond_mask = self.system1.condition(latent, latent_mask, image, state)
        head = self.system1.head
        if self.policy == "regression":
            return head(cond, cond_mask)
        x = noise

        if self.policy == "flow_matching":
            for i in range(self.steps):
                t = torch.full_like(x[:, 0, 0], i / self.steps)
                x = x + head(cond, cond_mask, x, t * 1000) / self.steps
            return x
        alpha_bar = self.system1.alpha_bar
        for i, k in enumerate(self.levels):
            ab = alpha_bar[k]
            v = head(cond, cond_mask, x, torch.full_like(x[:, 0, 0], float(k)))
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            ab_prev = alpha_bar[self.levels[i + 1]] if i + 1 < len(self.levels) else torch.ones((), device=x.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
