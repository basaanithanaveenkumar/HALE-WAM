"""ViTConfig: the built-in ViT, trained from scratch (``encoder: vit``)."""

from __future__ import annotations

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
