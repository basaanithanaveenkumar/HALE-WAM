"""DiffusionStrategy: DDPM with a cosine schedule, v-prediction and DDIM sampling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig

from .action_policy_strategy import ActionPolicyStrategy
from .system1_registry import SYSTEM1_POLICIES

#: Offset of the cosine schedule (Nichol & Dhariwal, 2021).
COSINE_OFFSET = 0.008
#: Lower clamp of ``alpha_bar`` so the first sampling step stays finite.
MIN_ALPHA_BAR = 1e-5


@SYSTEM1_POLICIES.register_module(name="diffusion")
class DiffusionStrategy(ActionPolicyStrategy):
    """DDPM (Ho et al., 2020) with a cosine schedule and DDIM sampling (eta = 0).

    The network predicts the velocity ``v = sqrt(a) eps - sqrt(1 - a) x0`` (Salimans & Ho,
    2022) rather than the noise: recovering ``x0 = sqrt(a) x_t - sqrt(1 - a) v`` needs no
    division by ``sqrt(a)``, so the first, near-pure-noise sampling step cannot amplify the
    prediction error (with noise prediction it is ~300x at ``a = 1e-5``).

    Complexity: sampling is O(steps) head evaluations; the noise levels are resolved once per
    ``steps`` value and cached.

    Args:
        config: dual-system settings (``diffusion_train_steps``).
    """

    def __init__(self, config: DualSystemConfig) -> None:
        super().__init__(config)
        steps = config.diffusion_train_steps
        x = torch.linspace(0, steps, steps + 1) / steps
        alpha_bar = torch.cos((x + COSINE_OFFSET) / (1 + COSINE_OFFSET) * math.pi / 2) ** 2
        alpha_bar = (alpha_bar / alpha_bar[0]).clamp(MIN_ALPHA_BAR, 1.0)
        # Not persistent: derived from the config, so checkpoints stay free of it.
        self.register_buffer("alpha_bar", alpha_bar[1:], persistent=False)
        self._levels: dict[int, list[int]] = {}

    def levels(self, steps: int) -> list[int]:
        """Evenly spaced training noise levels visited by DDIM, from noisiest to clean."""
        if steps not in self._levels:
            n = self.alpha_bar.numel()
            self._levels[steps] = torch.linspace(n - 1, 0, steps).round().long().tolist()
        return self._levels[steps]

    def prepare(self, steps: int) -> None:
        """Resolve the noise levels eagerly so tracing sees Python constants."""
        self.levels(steps)

    def loss(self, head: nn.Module, cond: torch.Tensor, cond_mask: torch.Tensor, actions: torch.Tensor,
             weight: torch.Tensor) -> torch.Tensor:
        """Mean squared error of the predicted velocity at a random noise level."""
        b = actions.size(0)
        noise = torch.randn_like(actions)
        k = torch.randint(0, self.alpha_bar.numel(), (b,), device=actions.device)
        ab = self.alpha_bar[k].view(b, 1, 1)
        noisy = ab.sqrt() * actions + (1 - ab).sqrt() * noise
        target = ab.sqrt() * noise - (1 - ab).sqrt() * actions  # v-prediction
        return self.masked_mean((head(cond, cond_mask, noisy, k) - target) ** 2, weight)

    def sample_from_noise(self, head: nn.Module, cond: torch.Tensor, cond_mask: torch.Tensor,
                          noise: torch.Tensor | None, steps: int) -> torch.Tensor:
        """DDIM over an evenly spaced subset of the training noise levels."""
        x = noise
        levels = self.levels(steps)
        for i, k in enumerate(levels):
            ab = self.alpha_bar[k]
            v = head(cond, cond_mask, x, torch.full_like(x[:, 0, 0], float(k)))
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            ab_prev = self.alpha_bar[levels[i + 1]] if i + 1 < len(levels) else torch.ones((), device=x.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
