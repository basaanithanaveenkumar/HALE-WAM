"""LanguageLatent: last-layer states of the instruction tokens only (DexVLA's language latent)."""

from __future__ import annotations

from .latent_extractor import LatentExtractor
from .sequence_slices import _text_slice


class LanguageLatent(LatentExtractor):
    """Last-layer states of the instruction tokens only (DexVLA's language latent)."""

    def forward(self, enc):
        return enc["hidden"][:, _text_slice(enc)], enc["mask"][:, _text_slice(enc)]
