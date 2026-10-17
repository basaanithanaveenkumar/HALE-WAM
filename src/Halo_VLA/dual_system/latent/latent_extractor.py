"""LatentExtractor: base class; subclasses implement :meth:`forward` for one latent type."""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentExtractor(nn.Module):
    """Base class; subclasses implement :meth:`forward` for one latent type."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, enc: dict) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError
