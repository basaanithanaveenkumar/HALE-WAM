"""Registry of latent extractors selectable by ``dual_system.latent``, and the factory."""

from __future__ import annotations

from exceptions import HaloVLAConfigError
from hale_core.registry import NamedRegistry

from dual_system.config import DualSystemConfig

from .act_token_latent import ActTokenLatent
from .language_latent import LanguageLatent
from .last_hidden_latent import LastHiddenLatent
from .latent_extractor import LatentExtractor
from .middle_layer_latent import MiddleLayerLatent
from .pooled_latent import PooledLatent

#: ``dual_system.latent`` -> builder ``(config, emb_dim, layer_width) -> LatentExtractor``.
LATENT_EXTRACTORS = NamedRegistry("latent extractor")

LATENT_EXTRACTORS.register_module(
    name="act_token", module=lambda config, emb_dim, layer_width: ActTokenLatent(emb_dim))
LATENT_EXTRACTORS.register_module(
    name="multi_act",
    module=lambda config, emb_dim, layer_width: ActTokenLatent(
        emb_dim, with_language=config.multi_act_with_language),
)
LATENT_EXTRACTORS.register_module(
    name="last_hidden", module=lambda config, emb_dim, layer_width: LastHiddenLatent(emb_dim))
LATENT_EXTRACTORS.register_module(
    name="language", module=lambda config, emb_dim, layer_width: LanguageLatent(emb_dim))
LATENT_EXTRACTORS.register_module(
    name="pooled", module=lambda config, emb_dim, layer_width: PooledLatent(emb_dim))
LATENT_EXTRACTORS.register_module(
    name="middle_layer", module=lambda config, emb_dim, layer_width: MiddleLayerLatent(layer_width))


def build_latent_extractor(config: DualSystemConfig, emb_dim: int, layer_width: int) -> LatentExtractor:
    """The extractor for ``config.latent`` (``middle_layer`` uses the decoder's raw width).

    Args:
        config: dual-system settings.
        emb_dim: the VLA's embedding width.
        layer_width: width of the decoder layer ``middle_layer`` taps.

    Raises:
        HaloVLAConfigError: for an unregistered latent type.
    """
    if config.latent not in LATENT_EXTRACTORS:
        raise HaloVLAConfigError(
            f"unknown dual_system.latent {config.latent!r}; choose one of {list(LATENT_EXTRACTORS)}"
        )
    return LATENT_EXTRACTORS.get(config.latent)(config, emb_dim, layer_width)
