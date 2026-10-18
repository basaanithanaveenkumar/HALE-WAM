"""LanguageLatent: last-layer states of the instruction tokens only (DexVLA's language latent)."""

from __future__ import annotations

from typing import Any

import torch

from .common import _text_slice
from .latent_extractor import LatentExtractor
from .latent_registry import LATENT_EXTRACTORS


@LATENT_EXTRACTORS.register_module(name="language")
class LanguageLatent(LatentExtractor):
    """Last-layer states of the instruction tokens only (DexVLA's language latent)."""

    def forward(self, enc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """The instruction tokens' states and mask."""
        return enc["hidden"][:, _text_slice(enc)], enc["mask"][:, _text_slice(enc)]
