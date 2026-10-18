"""ActTokenLatent: final states of System 2's appended ``<ACT>`` queries."""

from __future__ import annotations

from typing import Any

import torch

from dual_system.config import DualSystemConfig

from .common import _text_slice
from .latent_extractor import LatentExtractor
from .latent_registry import LATENT_EXTRACTORS


def _act_slice(enc: dict[str, Any]) -> slice:
    """Positions of the ``<ACT>`` queries (after the prefix and the instruction)."""
    start = enc["num_prefix"] + enc["text_len"]
    return slice(start, start + enc["num_extra"])


@LATENT_EXTRACTORS.register_module(name=["act_token", "multi_act"])
class ActTokenLatent(LatentExtractor):
    """Final states of the appended ``<ACT>`` queries (LCB, OpenHelix: 1 query; RoboDual: K).

    Args:
        width: token width.
        with_language: also pass the instruction's states (RoboDual's task + action latent).
    """

    def __init__(self, width: int, with_language: bool = False) -> None:
        super().__init__(width)
        self.with_language = with_language

    @staticmethod
    def _with_language(config: DualSystemConfig) -> bool:
        return config.latent == "multi_act" and config.multi_act_with_language

    @classmethod
    def from_config(cls, config: DualSystemConfig, emb_dim: int, layer_width: int) -> ActTokenLatent:
        """``multi_act`` with ``multi_act_with_language`` also passes the instruction states."""
        return cls(emb_dim, with_language=cls._with_language(config))

    @classmethod
    def static_length(cls, config: DualSystemConfig) -> int | None:
        """``num_act_tokens`` queries, unless the instruction states are appended."""
        return None if cls._with_language(config) else config.num_act_tokens

    def forward(self, enc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``<ACT>`` states (plus the instruction states when ``with_language``)."""
        tokens = enc["hidden"][:, _act_slice(enc)]
        mask = enc["mask"][:, _act_slice(enc)]
        if self.with_language:  # RoboDual: task latent (instruction states) + action latent
            text = enc["hidden"][:, _text_slice(enc)]
            text_mask = enc["mask"][:, _text_slice(enc)]
            tokens, mask = torch.cat([tokens, text], 1), torch.cat([mask, text_mask], 1)
        return tokens, mask
