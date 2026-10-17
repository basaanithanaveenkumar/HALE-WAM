"""Helpers shared by the ``vision_encoders`` package."""

from __future__ import annotations

import torch
import torch.nn as nn
from exceptions import HaloVLAConfigError

from .hf_vision_encoder import HFVisionEncoder
from .vision_encoder_info import VisionEncoderInfo

#: Default Hub repos for the pretrained encoders (both checked on the Hub).
DEFAULT_HF_ENCODERS = {
    "siglip": "google/siglip-base-patch16-224",  # Zhai et al., 2023, arXiv:2303.15343
    "clip": "openai/clip-vit-base-patch16",      # Radford et al., 2021, arXiv:2103.00020
}


ENCODER_TYPES = ("vit", "siglip", "clip", "local")


def build_vision_encoder(config) -> tuple[nn.Module, VisionEncoderInfo]:
    """The encoder ``config.vision_encoder`` selects, plus its token width and count.

    * ``vit`` (default): the built-in ``VisTransformer`` - trained from scratch, ``emb_dim`` wide.
    * ``siglip`` / ``clip``: pretrained Hub towers (frozen by default).
    * ``local``: any ``hale_core.nn.vision`` backbone by name (frozen by default; optional
      ``vision_freeze_up_to_stage`` and ``vision_pretrained_path``).

    The token count is measured with one dummy forward pass, so it is exact for every
    architecture (patch tokens for ViTs, the finest feature-map grid for CNNs).
    """
    kind = config.vision_encoder
    if kind not in ENCODER_TYPES:
        raise HaloVLAConfigError(f"vision_encoder must be one of {ENCODER_TYPES}, got {kind!r}")
    freeze = config.resolved_vision_freeze
    if kind == "vit":
        from models.vit import VisTransformer

        encoder: nn.Module = VisTransformer(
            img_size=config.img_size, p_size=config.patch_size, in_chans=config.in_chans,
            emb_dim=config.emb_dim, num_layers=config.vit_num_layers, num_heads=config.vit_num_heads,
            mlp_dim=config.vit_mlp_dim, drop_fact=config.vit_drop,
        )
        if freeze:
            encoder.requires_grad_(False)
        return encoder, VisionEncoderInfo(config.emb_dim, (config.img_size // config.patch_size) ** 2)
    if kind in ("siglip", "clip"):
        model_id = config.vision_model_id or DEFAULT_HF_ENCODERS[kind]
        encoder = HFVisionEncoder(kind, model_id, config.img_size, freeze)
    else:
        from models.vision_backbone import HaloVisionBackbone

        encoder = HaloVisionBackbone(
            backbone_name=config.vision_backbone, img_size=config.img_size, freeze_encoder=freeze,
            freeze_up_to_stage=config.vision_freeze_up_to_stage,
            pretrained_path=config.vision_pretrained_path,
        )
    with torch.no_grad():
        was_training = encoder.training
        encoder.eval()
        tokens = encoder(torch.zeros(1, config.in_chans, config.img_size, config.img_size))
        encoder.train(was_training)
    return encoder, VisionEncoderInfo(int(tokens.shape[-1]), int(tokens.shape[1]))
