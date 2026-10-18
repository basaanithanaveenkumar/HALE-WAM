"""LastHiddenLatent: every last-layer state, vision and language (DP-VLA)."""

from __future__ import annotations

from typing import Any

import torch

from .latent_extractor import LatentExtractor
from .latent_registry import LATENT_EXTRACTORS


@LATENT_EXTRACTORS.register_module(name="last_hidden")
class LastHiddenLatent(LatentExtractor):
    """Every last-layer state, vision and language (DP-VLA)."""

    def forward(self, enc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """All hidden states and the sequence mask."""
        return enc["hidden"], enc["mask"]
