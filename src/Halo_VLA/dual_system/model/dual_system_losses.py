"""DualSystemLosses: loss terms of one training step (``total`` is back-propagated)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DualSystemLosses:
    """Loss terms of one training step (``total`` is back-propagated)."""

    total: torch.Tensor
    policy: torch.Tensor
    aux: torch.Tensor
    align: torch.Tensor
    lm: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
