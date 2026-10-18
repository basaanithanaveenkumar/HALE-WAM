"""RegressionStrategy: predict the chunk directly with a masked L1 (DP-VLA, HiRT / RT-1)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .action_policy_strategy import ActionPolicyStrategy
from .system1_registry import SYSTEM1_POLICIES


@SYSTEM1_POLICIES.register_module(name="regression")
class RegressionStrategy(ActionPolicyStrategy):
    """One-shot regression of the action chunk (no noise, no sampler loop)."""

    needs_noise = False
    has_sampler_step = False

    def loss(self, head: nn.Module, cond: torch.Tensor, cond_mask: torch.Tensor, actions: torch.Tensor,
             weight: torch.Tensor) -> torch.Tensor:
        """Masked mean L1 between the predicted and the target chunk."""
        return self.masked_mean((head(cond, cond_mask) - actions).abs(), weight)

    def sample_from_noise(self, head: nn.Module, cond: torch.Tensor, cond_mask: torch.Tensor,
                          noise: torch.Tensor | None, steps: int) -> torch.Tensor:
        """The head's prediction; ``noise`` and ``steps`` are unused."""
        return head(cond, cond_mask)
