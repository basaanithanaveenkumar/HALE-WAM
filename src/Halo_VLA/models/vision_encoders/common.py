"""build_vision_encoder: the encoder ``HaloVLMConfig.vision_encoder`` selects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from exceptions import HaloVLAConfigError
from hale_core.exceptions import RegistryError

from . import encoder_builders  # noqa: F401 - registers the built-in encoders
from .encoder_builders import DEFAULT_HF_ENCODERS
from .encoder_registry import VISION_ENCODERS
from .vision_encoder_info import VisionEncoderInfo

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig

#: Registered ``vision_encoder`` names, in registration order.
ENCODER_TYPES = tuple(VISION_ENCODERS.items)


def build_vision_encoder(config: HaloVLMConfig) -> tuple[nn.Module, VisionEncoderInfo]:
    """The encoder ``config.vision_encoder`` selects, plus its token width and count.

    * ``vit`` (default): the built-in ``VisTransformer`` - trained from scratch, ``emb_dim`` wide.
    * ``siglip`` / ``clip``: pretrained Hub towers (frozen by default).
    * ``local``: any ``hale_core.nn.vision`` backbone by name (frozen by default; optional
      ``vision_freeze_up_to_stage`` and ``vision_pretrained_path``).

    The token count is measured with one dummy forward pass (analytic for ``vit``), so it is
    exact for every architecture (patch tokens for ViTs, the finest feature-map grid for CNNs).
    New kinds register a builder in :data:`VISION_ENCODERS`.

    Args:
        config: model settings.

    Returns:
        ``(encoder, info)``.

    Raises:
        HaloVLAConfigError: for an unregistered ``vision_encoder``.
    """
    kind = config.vision_encoder
    try:
        builder = VISION_ENCODERS.get(kind)
    except RegistryError as err:
        raise HaloVLAConfigError(f"vision_encoder must be one of {ENCODER_TYPES}, got {kind!r}") from err
    encoder, info = builder(config)
    if info is not None:
        return encoder, info
    with torch.no_grad():
        was_training = encoder.training
        encoder.eval()
        tokens = encoder(torch.zeros(1, config.in_chans, config.img_size, config.img_size))
        encoder.train(was_training)
    return encoder, VisionEncoderInfo(int(tokens.shape[-1]), int(tokens.shape[1]))
