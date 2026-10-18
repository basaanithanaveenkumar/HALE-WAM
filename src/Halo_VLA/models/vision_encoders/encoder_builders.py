"""Builders that register each vision encoder kind in :data:`VISION_ENCODERS`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from .encoder_registry import VISION_ENCODERS
from .hf_vision_encoder import HFVisionEncoder
from .vision_encoder_info import VisionEncoderInfo

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig

#: A builder's result: the encoder and its info, or ``None`` to have the info measured.
EncoderBuild = tuple[nn.Module, VisionEncoderInfo | None]

#: Default Hub repos for the pretrained encoders (both checked on the Hub).
DEFAULT_HF_ENCODERS = {
    "siglip": "google/siglip-base-patch16-224",  # Zhai et al., 2023, arXiv:2303.15343
    "clip": "openai/clip-vit-base-patch16",  # Radford et al., 2021, arXiv:2103.00020
}


@VISION_ENCODERS.register_module(name="vit")
def build_vit_encoder(config: HaloVLMConfig) -> EncoderBuild:
    """The built-in ``VisTransformer``: trained from scratch, ``emb_dim`` wide."""
    from models.vit import VisTransformer  # lazy: keeps the registry importable on its own

    encoder = VisTransformer(
        img_size=config.img_size,
        p_size=config.patch_size,
        in_chans=config.in_chans,
        emb_dim=config.emb_dim,
        num_layers=config.vit_num_layers,
        num_heads=config.vit_num_heads,
        mlp_dim=config.vit_mlp_dim,
        drop_fact=config.vit_drop,
    )
    if config.resolved_vision_freeze:
        encoder.requires_grad_(False)
    return encoder, VisionEncoderInfo(config.emb_dim, (config.img_size // config.patch_size) ** 2)


def _build_hub_encoder(kind: str, config: HaloVLMConfig) -> EncoderBuild:
    """A pretrained SigLIP / CLIP tower (frozen by default)."""
    model_id = config.vision_model_id or DEFAULT_HF_ENCODERS[kind]
    return HFVisionEncoder(kind, model_id, config.img_size, config.resolved_vision_freeze), None


@VISION_ENCODERS.register_module(name="siglip")
def build_siglip_encoder(config: HaloVLMConfig) -> EncoderBuild:
    """Pretrained SigLIP vision tower."""
    return _build_hub_encoder("siglip", config)


@VISION_ENCODERS.register_module(name="clip")
def build_clip_encoder(config: HaloVLMConfig) -> EncoderBuild:
    """Pretrained CLIP vision tower."""
    return _build_hub_encoder("clip", config)


@VISION_ENCODERS.register_module(name="local")
def build_local_encoder(config: HaloVLMConfig) -> EncoderBuild:
    """Any ``hale_core.nn.vision`` backbone by name (optional stage freezing and checkpoint)."""
    from models.vision_backbone import HaloVisionBackbone  # lazy: pulls in the HaleBlocks backbones

    encoder = HaloVisionBackbone(
        backbone_name=config.vision_backbone,
        img_size=config.img_size,
        freeze_encoder=config.resolved_vision_freeze,
        freeze_up_to_stage=config.vision_freeze_up_to_stage,
        pretrained_path=config.vision_pretrained_path,
    )
    return encoder, None
