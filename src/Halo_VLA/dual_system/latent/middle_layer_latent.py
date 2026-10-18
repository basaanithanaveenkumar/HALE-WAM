"""MiddleLayerLatent: states of a middle decoder layer (GR00T-N1 style: more visual detail, cheaper)."""

from __future__ import annotations

from typing import Any

import torch

from dual_system.config import DualSystemConfig
from exceptions import ModelStateError

from .latent_extractor import LatentExtractor
from .latent_registry import LATENT_EXTRACTORS


@LATENT_EXTRACTORS.register_module(name="middle_layer")
class MiddleLayerLatent(LatentExtractor):
    """States of a middle decoder layer (GR00T-N1 style: more visual detail, cheaper)."""

    needs_layer_capture = True

    @classmethod
    def from_config(
        cls, config: DualSystemConfig, emb_dim: int, layer_width: int
    ) -> MiddleLayerLatent:
        """The latent lives in the decoder's raw width, not ``emb_dim``."""
        return cls(layer_width)

    def forward(self, enc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """The captured layer's states and the sequence mask.

        Raises:
            ModelStateError: if System 2 did not capture a layer.
        """
        if enc["layer_hidden"] is None:
            raise ModelStateError("middle_layer latent needs System 2 to capture a layer")
        return enc["layer_hidden"], enc["mask"]
