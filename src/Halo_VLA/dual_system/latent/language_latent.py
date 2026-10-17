"""LanguageLatent: last-layer states of the instruction tokens only (DexVLA's language latent)."""

from __future__ import annotations

from .latent_extractor import LatentExtractor


class LanguageLatent(LatentExtractor):
    """Last-layer states of the instruction tokens only (DexVLA's language latent)."""

    def forward(self, enc):
        from .common import _text_slice  # lazy: avoids an import cycle
        return enc["hidden"][:, _text_slice(enc)], enc["mask"][:, _text_slice(enc)]
