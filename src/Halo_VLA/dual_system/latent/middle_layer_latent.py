"""MiddleLayerLatent: states of a middle decoder layer (GR00T-N1 style: more visual detail, cheaper)."""

from __future__ import annotations

from exceptions import ModelStateError

from .latent_extractor import LatentExtractor


class MiddleLayerLatent(LatentExtractor):
    """States of a middle decoder layer (GR00T-N1 style: more visual detail, cheaper)."""

    def forward(self, enc):
        if enc["layer_hidden"] is None:
            raise ModelStateError("middle_layer latent needs System 2 to capture a layer")
        return enc["layer_hidden"], enc["mask"]
