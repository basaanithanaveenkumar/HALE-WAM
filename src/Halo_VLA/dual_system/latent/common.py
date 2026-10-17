"""Helpers shared by the ``latent`` package."""

from __future__ import annotations

from dual_system.config import DualSystemConfig

from .last_hidden_latent import LastHiddenLatent
from .latent_extractor import LatentExtractor
from .middle_layer_latent import MiddleLayerLatent
from .pooled_latent import PooledLatent


def _text_slice(enc: dict) -> slice:
    return slice(enc["num_prefix"], enc["num_prefix"] + enc["text_len"])


def build_latent_extractor(config: DualSystemConfig, emb_dim: int, layer_width: int) -> LatentExtractor:
    """The extractor for ``config.latent``."""
    from .act_token_latent import ActTokenLatent  # lazy: avoids an import cycle
    from .language_latent import LanguageLatent  # lazy: avoids an import cycle
    kind = config.latent
    if kind in ("act_token", "multi_act"):
        return ActTokenLatent(emb_dim, with_language=kind == "multi_act" and config.multi_act_with_language)
    if kind == "last_hidden":
        return LastHiddenLatent(emb_dim)
    if kind == "language":
        return LanguageLatent(emb_dim)
    if kind == "pooled":
        return PooledLatent(emb_dim)
    return MiddleLayerLatent(layer_width)
