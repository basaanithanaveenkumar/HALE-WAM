"""DecoderConfig: the built-in decoder transformer (unused when ``model.llm.backbone`` is set)."""

from __future__ import annotations

from config.sections.base import VLASection


class DecoderConfig(VLASection):
    """The built-in decoder transformer (unused when ``model.llm.backbone`` is set)."""

    OVERRIDES = {
        "num_layers": "dec_num_layers",
        "num_heads": "dec_num_heads",
        "mlp_dim": "dec_mlp_dim",
        "dropout": "dec_drop",
    }

    num_layers: int = 8
    num_heads: int = 16
    mlp_dim: int = 512
    dropout: float = 0.0
