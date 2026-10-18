"""build_image_projector: the projector ``HaloVLMConfig.proj_type`` selects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from exceptions import HaloVLAConfigError
from hale_core.exceptions import RegistryError

from . import projector_builders  # noqa: F401 - registers the built-in projectors
from .projector_registry import IMAGE_PROJECTORS

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig


def build_image_projector(config: "HaloVLMConfig") -> nn.Module:
    """Build the vision -> decoder projector selected by ``config.proj_type``.

    * ``"mlp"``: :class:`ImageProjector`, one token per ViT patch.
    * ``"qformer"``: :class:`QFormer`, ``config.qformer_num_queries`` tokens per image.
    * ``"gated_cross_attention"``: :class:`GatedCrossAttentionProjector`,
      ``config.gated_xattn_num_latents`` tokens per image.

    New kinds register a builder in :data:`IMAGE_PROJECTORS`.

    Args:
        config: model settings.

    Returns:
        The projector module.

    Raises:
        HaloVLAConfigError: for an unregistered ``proj_type``.
    """
    try:
        builder = IMAGE_PROJECTORS.get(config.proj_type)
    except RegistryError as err:
        raise HaloVLAConfigError(
            f"Unknown proj_type {config.proj_type!r}; expected one of {tuple(IMAGE_PROJECTORS.items)}"
        ) from err
    return builder(config)
