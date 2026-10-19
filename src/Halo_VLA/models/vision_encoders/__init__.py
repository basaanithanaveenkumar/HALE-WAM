"""Configurable vision encoder for Halo-VLA: built-in ViT, SigLIP / CLIP, or any HaleBlocks backbone.

Mirrors Hale-VLM's ``vision.encoder: siglip | clip | local`` so both models can use the same
pretrained towers. Every encoder maps ``(B, 3, H, W)`` images (normalised with the datasets'
ImageNet statistics) to ``(B, N, D)`` tokens; the image projector then maps ``D -> emb_dim``."""

from .vision_encoder_registry import (  # noqa: F401
    DEFAULT_HF_ENCODERS,
    ENCODER_TYPES,
    VISION_ENCODERS,
    build_vision_encoder,
)
from .hf_vision_encoder import DATASET_MEAN, DATASET_STD, HFVisionEncoder  # noqa: F401
from .vision_encoder_info import VisionEncoderInfo  # noqa: F401

__all__ = [
    "DATASET_MEAN",
    "DATASET_STD",
    "DEFAULT_HF_ENCODERS",
    "ENCODER_TYPES",
    "VISION_ENCODERS",
    "HFVisionEncoder",
    "VisionEncoderInfo",
    "build_vision_encoder",
]
