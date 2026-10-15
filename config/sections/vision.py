"""``model.vision``: vision encoder and the vision -> decoder projector (Hale-VLM layout)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from config.sections.base import VLASection


class ViTConfig(VLASection):
    """The built-in ViT, trained from scratch (``encoder: vit``)."""

    OVERRIDES = {"patch_size": "patch_size", "num_layers": "vit_num_layers", "num_heads": "vit_num_heads",
                 "mlp_dim": "vit_mlp_dim", "dropout": "vit_drop"}

    patch_size: int = 16
    num_layers: int = 6
    num_heads: int = 16
    mlp_dim: int = 512
    dropout: float = 0.0


class QFormerConfig(VLASection):
    """BLIP-2 Q-Former settings, used when ``projector_type: qformer``."""

    OVERRIDES = {"num_queries": "qformer_num_queries", "hidden_dim": "qformer_hidden_dim",
                 "num_layers": "qformer_num_layers", "num_heads": "qformer_num_heads",
                 "cross_attention_freq": "qformer_cross_attention_freq", "ffn_mult": "qformer_ffn_mult",
                 "dropout": "qformer_dropout"}

    num_queries: int = 32
    hidden_dim: int | None = None          # None -> min(projector_llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    cross_attention_freq: int = 1          # cross-attend to the image in every N-th layer (BLIP-2: 2)
    ffn_mult: int = 4
    dropout: float = 0.0


class GatedCrossAttentionConfig(VLASection):
    """Flamingo-style gated cross-attention, used when ``projector_type: gated_cross_attention``."""

    OVERRIDES = {"num_latents": "gated_xattn_num_latents", "hidden_dim": "gated_xattn_hidden_dim",
                 "num_layers": "gated_xattn_num_layers", "num_heads": "gated_xattn_num_heads",
                 "ffn_mult": "gated_xattn_ffn_mult", "dropout": "gated_xattn_dropout",
                 "gate_init": "gated_xattn_gate_init"}

    num_latents: int = 32
    hidden_dim: int | None = None
    num_layers: int = 2
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    gate_init: float = 1.0                 # 0.0 = Flamingo identity start; >0 lets images flow at step 0


class LocalBackboneConfig(VLASection):
    """A ``hale_core.nn.vision`` backbone (``encoder: local``)."""

    ARGS = {"backbone_name": "vision_backbone", "freeze_up_to_stage": "vision_freeze_up_to_stage",
            "pretrained_path": "vision_pretrained_path"}

    backbone_name: str = "dino_vit_small"
    freeze_up_to_stage: int = -1           # freeze stages 0..N, fine-tune the rest
    pretrained_path: str | None = None


class VisionConfig(VLASection):
    """Vision tower and projector.

    ``encoder``: ``vit`` (built-in, from scratch), ``siglip`` / ``clip`` (pretrained Hub
    towers, ``model_id``), ``local`` (any HaleBlocks vision backbone, ``local_backbone``).
    """

    ARGS = {"encoder": "vision_encoder", "model_id": "vision_model_id", "freeze_encoder": "vision_freeze",
            "projector_type": "projector_type"}
    OVERRIDES = {"image_size": "img_size", "projector_vision_dim": "proj_vision_dim",
                 "projector_llm_dim": "proj_llm_dim"}

    encoder: Literal["vit", "siglip", "clip", "local"] = "vit"
    model_id: str | None = None            # siglip / clip repo (default: *-base-patch16-224)
    image_size: int = 224
    freeze_encoder: bool | None = None     # None: frozen for pretrained encoders, trained for vit
    projector_type: Literal["mlp", "qformer", "gated_cross_attention"] = "mlp"
    projector_vision_dim: int | None = None
    projector_llm_dim: int | None = None
    vit: ViTConfig = Field(default_factory=ViTConfig)
    qformer: QFormerConfig = Field(default_factory=QFormerConfig)
    gated_cross_attention: GatedCrossAttentionConfig = Field(default_factory=GatedCrossAttentionConfig)
    local_backbone: LocalBackboneConfig = Field(default_factory=LocalBackboneConfig)
