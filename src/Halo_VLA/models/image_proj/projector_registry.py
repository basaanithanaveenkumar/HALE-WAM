"""Registry of vision -> decoder projectors, keyed by ``HaloVLMConfig.proj_type``."""

from __future__ import annotations

from hale_core.registry import NamedRegistry

#: ``proj_type`` name -> ``(config) -> nn.Module`` builder.
IMAGE_PROJECTORS = NamedRegistry("image projector")

__all__ = ["IMAGE_PROJECTORS"]
