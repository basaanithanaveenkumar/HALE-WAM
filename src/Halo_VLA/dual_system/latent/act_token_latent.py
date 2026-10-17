"""ActTokenLatent: actTokenLatent."""

from __future__ import annotations

import torch

from .latent_extractor import LatentExtractor


def _act_slice(enc: dict) -> slice:
    start = enc["num_prefix"] + enc["text_len"]
    return slice(start, start + enc["num_extra"])


class ActTokenLatent(LatentExtractor):
    """Final states of the appended ``<ACT>`` queries (LCB, OpenHelix: 1 query; RoboDual: K)."""

    def __init__(self, width: int, with_language: bool = False) -> None:
        super().__init__(width)
        self.with_language = with_language

    def forward(self, enc):
        from .common import _text_slice  # lazy: avoids an import cycle
        tokens = enc["hidden"][:, _act_slice(enc)]
        mask = enc["mask"][:, _act_slice(enc)]
        if self.with_language:  # RoboDual: task latent (instruction states) + action latent
            text = enc["hidden"][:, _text_slice(enc)]
            text_mask = enc["mask"][:, _text_slice(enc)]
            tokens, mask = torch.cat([tokens, text], 1), torch.cat([mask, text_mask], 1)
        return tokens, mask
