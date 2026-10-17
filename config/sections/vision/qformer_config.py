"""QFormerConfig: bLIP-2 Q-Former settings, used when ``projector_type: qformer``."""

from __future__ import annotations

from config.sections.base import VLASection


class QFormerConfig(VLASection):
    """BLIP-2 Q-Former settings, used when ``projector_type: qformer``."""

    OVERRIDES = {"num_queries": "qformer_num_queries", "hidden_dim": "qformer_hidden_dim",
                 "num_layers": "qformer_num_layers", "num_heads": "qformer_num_heads",
                 "cross_attention_freq": "qformer_cross_attention_freq", "ffn_mult": "qformer_ffn_mult",
                 "dropout": "qformer_dropout"}

    num_queries: int = 32
    hidden_dim: int | None = None          # None -> min(projector_llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    cross_attention_freq: int = 1          # cross-attend to the image in every N-th layer (BLIP-2: 2)
    ffn_mult: int = 4
    dropout: float = 0.0
