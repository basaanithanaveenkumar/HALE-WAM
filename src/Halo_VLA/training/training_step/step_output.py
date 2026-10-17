"""StepOutput: losses of one micro-step (``total`` is what gets back-propagated)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class StepOutput:
    """Losses of one micro-step (``total`` is what gets back-propagated)."""

    total: torch.Tensor
    lang: torch.Tensor
    act: torch.Tensor
    vis: torch.Tensor
    vis_sub: dict[str, torch.Tensor] = field(default_factory=dict)
    action_hiddens: torch.Tensor | None = None
    """Decoder states at ``<halo_action>`` positions (evaluation samples actions from them)."""
