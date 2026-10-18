"""VisionConfig: vision tower and projector."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from config.sections.base import VLASection

from .gated_cross_attention_config import GatedCrossAttentionConfig
from .local_backbone_config import LocalBackboneConfig
from .qformer_config import QFormerConfig
from .vit_config import ViTConfig


class VisionConfig(VLASection):
    """Vision tower and projector.

    ``encoder``: ``vit`` (built-in, from scratch), ``siglip`` / ``clip`` (pretrained Hub
    towers, ``model_id``), ``local`` (any HaleBlocks vision backbone, ``local_backbone``).
    """

    ARGS = {
        "encoder": "vision_encoder",
        "model_id": "vision_model_id",
        "freeze_encoder": "vision_freeze",
        "projector_type": "projector_type",
    }
    OVERRIDES = {
        "image_size": "img_size",
        "projector_vision_dim": "proj_vision_dim",
        "projector_llm_dim": "proj_llm_dim",
    }

    encoder: Literal["vit", "siglip", "clip", "local"] = "vit"
    model_id: str | None = None  # siglip / clip repo (default: *-base-patch16-224)
    image_size: int = 224
    freeze_encoder: bool | None = None  # None: frozen for pretrained encoders, trained for vit
    projector_type: Literal["mlp", "qformer", "gated_cross_attention"] = "mlp"
    projector_vision_dim: int | None = None
    projector_llm_dim: int | None = None
    vit: ViTConfig = Field(default_factory=ViTConfig)
    qformer: QFormerConfig = Field(default_factory=QFormerConfig)
    gated_cross_attention: GatedCrossAttentionConfig = Field(
        default_factory=GatedCrossAttentionConfig
    )
    local_backbone: LocalBackboneConfig = Field(default_factory=LocalBackboneConfig)
