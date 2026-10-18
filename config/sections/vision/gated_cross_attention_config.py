"""GatedCrossAttentionConfig: flamingo-style gated cross-attention, used when ``projector_type: gated_cross_attention``."""

from __future__ import annotations

from config.sections.base import VLASection


class GatedCrossAttentionConfig(VLASection):
    """Flamingo-style gated cross-attention, used when ``projector_type: gated_cross_attention``."""

    OVERRIDES = {
        "num_latents": "gated_xattn_num_latents",
        "hidden_dim": "gated_xattn_hidden_dim",
        "num_layers": "gated_xattn_num_layers",
        "num_heads": "gated_xattn_num_heads",
        "ffn_mult": "gated_xattn_ffn_mult",
        "dropout": "gated_xattn_dropout",
        "gate_init": "gated_xattn_gate_init",
    }

    num_latents: int = 32
    hidden_dim: int | None = None
    num_layers: int = 2
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    gate_init: float = 1.0  # 0.0 = Flamingo identity start; >0 lets images flow at step 0
