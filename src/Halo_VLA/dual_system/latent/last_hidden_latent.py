"""LastHiddenLatent: every last-layer state, vision and language (DP-VLA)."""

from __future__ import annotations

from .latent_extractor import LatentExtractor


class LastHiddenLatent(LatentExtractor):
    """Every last-layer state, vision and language (DP-VLA)."""

    def forward(self, enc):
        return enc["hidden"], enc["mask"]
