"""Helpers shared by the ``latent`` package."""

from __future__ import annotations

from typing import Any

from dual_system.config import DualSystemConfig

from .latent_extractor import LatentExtractor
from .latent_registry import LATENT_EXTRACTORS


def _text_slice(enc: dict[str, Any]) -> slice:
    """Positions of the instruction tokens in System 2's sequence."""
    return slice(enc["num_prefix"], enc["num_prefix"] + enc["text_len"])


def build_latent_extractor(
    config: DualSystemConfig, emb_dim: int, layer_width: int
) -> LatentExtractor:
    """Build the extractor registered for ``config.latent``.

    Args:
        config: dual-system settings.
        emb_dim: System 2's embedding width.
        layer_width: the decoder's raw hidden width (``middle_layer`` latents).

    Returns:
        The extractor module.

    Raises:
        hale_core.exceptions.RegistryError: if ``config.latent`` has no registered extractor.
    """
    return LATENT_EXTRACTORS.get(config.latent).from_config(config, emb_dim, layer_width)
