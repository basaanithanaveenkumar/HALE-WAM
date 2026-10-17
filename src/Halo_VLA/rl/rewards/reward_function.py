"""RewardFunction: anything callable that maps a batch of actions to a reward per sample."""

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
