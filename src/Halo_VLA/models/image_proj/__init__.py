"""Vision -> decoder projectors (MLP, Q-Former, gated cross-attention), registered by ``proj_type``."""

from . import projector_builders  # noqa: F401 - registers the built-in projectors
from .build_image_projector import build_image_projector
from .image_projector import ImageProjector
from .projector_registry import IMAGE_PROJECTORS

#: Registered ``proj_type`` names, in registration order.
PROJECTOR_TYPES = tuple(IMAGE_PROJECTORS.items)

__all__ = ["IMAGE_PROJECTORS", "PROJECTOR_TYPES", "ImageProjector", "build_image_projector"]
