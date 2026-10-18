"""Registry of vision encoders, keyed by ``HaloVLMConfig.vision_encoder``."""

from __future__ import annotations

from hale_core.registry import NamedRegistry

#: ``vision_encoder`` name -> ``(config) -> (nn.Module, VisionEncoderInfo | None)`` builder; the
#: module maps images to tokens and ``None`` info means "measure it with a dummy forward pass".
VISION_ENCODERS = NamedRegistry("vision encoder")

__all__ = ["VISION_ENCODERS"]
