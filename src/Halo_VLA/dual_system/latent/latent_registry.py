"""Registry of System 2 -> System 1 latent extractors, keyed by ``DualSystemConfig.latent``."""

from __future__ import annotations

from hale_core.registry import NamedRegistry

#: ``latent`` name -> :class:`~dual_system.latent.latent_extractor.LatentExtractor` subclass.
LATENT_EXTRACTORS = NamedRegistry("latent extractor")

__all__ = ["LATENT_EXTRACTORS"]
