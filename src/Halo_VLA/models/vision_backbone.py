"""
VisionBackbone adapter for Halo-VLA (HALE-WAM).

Role in the VLA pipeline
-------------------------
The Halo-VLA model processes robot observations as a sequence:
  image_obs → vision_backbone → image_features → [Q-Former / projection]
                                                         ↓
                            language_tokens + action_tokens → transformer → action

The existing vit.py provides a basic ViT implementation tightly coupled to
the repo's own TransformerBlock. This module provides an alternative: use
any production backbone from hale_core.nn.vision, with proper architecture
choices (ConvNeXt for fast inference, SwinV2 for dense perception,
DINOv2 ViT-G for best features, EVA-02 for CLIP-aligned embeddings).

Both single-frame and multi-frame inputs are supported:
  single-frame: [B, 3, H, W]
  multi-frame:  [B, T, 3, H, W] — frames processed independently, then stacked

Usage
-----
    from Halo_VLA.models.vision_backbone import HaloVisionBackbone

    backbone = HaloVisionBackbone(
        backbone_name="convnext_base",
        img_size=224,
        freeze_encoder=False,
    )
    features = backbone(images)            # [B, N, D] patch tokens
    features = backbone(video_frames)      # [B, T*N, D] for T frames

Config-driven:

    # In your VLA config dict:
    "vision": {
        "backbone_name": "swinv2_tiny",
        "img_size": 224,
        "freeze_encoder": True,
        "return_patch_tokens": True,
    }
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class HaloVisionBackbone(nn.Module):
    """Vision encoder for Halo-VLA using hale_core registered backbones.

    Outputs a sequence of visual tokens suitable for cross-attention with
    language and action tokens.

    Args:
        backbone_name:
            Any backbone registered in hale_core.nn.vision.
            Examples: "convnext_base", "vit_large", "swinv2_tiny",
            "dino_vit_giant", "eva_02_large", "resnet50".

        img_size:
            Expected input image resolution.

        freeze_encoder:
            If True, backbone weights are frozen (no gradient).

        freeze_up_to_stage:
            If >= 0, freeze backbone stages 0..N.

        pretrained_path:
            Optional path to local backbone checkpoint.

        return_patch_tokens:
            If True, return per-patch tokens [B, N, D] instead of [B, 1, D].
            Most VLA projectors (Q-Former, gated cross-attention) need tokens.

        proj_dim:
            If not None, project backbone output tokens to this dimension
            via a learned linear layer (useful for dimension matching with LLM).
            If None, tokens are returned at backbone's native dimension.
    """

    def __init__(
        self,
        backbone_name:       str          = "convnext_base",
        img_size:            int          = 224,
        freeze_encoder:      bool         = True,
        freeze_up_to_stage:  int          = -1,
        pretrained_path:     Optional[str] = None,
        return_patch_tokens: bool          = True,
        proj_dim:            Optional[int] = None,
    ) -> None:
        super().__init__()

        # ── Build backbone ────────────────────────────────────────────────────
        from hale_core.nn.vision import build_vision_backbone

        self.backbone = build_vision_backbone(
            backbone_name,
            img_size=img_size,
            in_chans=3,
        )
        self.backbone_name       = backbone_name
        self.return_patch_tokens = return_patch_tokens
        self._freeze_encoder     = freeze_encoder

        # Load custom weights if provided
        if pretrained_path:
            from hale_core.config.sections.backbone import _load_pretrained
            _load_pretrained(self.backbone, pretrained_path)

        # Freezing
        if freeze_encoder:
            self.backbone.freeze_all()
            self.backbone.eval()
        elif freeze_up_to_stage >= 0:
            self.backbone.freeze_up_to_stage(freeze_up_to_stage)

        # ── Optional projection layer ─────────────────────────────────────────
        # Projects backbone token dimension to a target LLM dimension.
        # E.g., ConvNeXt-base has embed_dim=1024; LLM might expect 2048 → use proj_dim=2048.
        self.proj: Optional[nn.Linear]
        if proj_dim is not None:
            self.proj = nn.Linear(self.backbone.embed_dim, proj_dim, bias=False)
            self._out_dim = proj_dim
        else:
            self.proj = None
            self._out_dim = self.backbone.embed_dim

    @property
    def output_dim(self) -> int:
        """Dimension of each output token."""
        return self._out_dim

    def train(self, mode: bool = True) -> "HaloVisionBackbone":
        super().train(mode)
        if self._freeze_encoder:
            self.backbone.eval()  # a frozen backbone keeps dropout off and BatchNorm statistics fixed
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Encode images/video frames into visual token sequences.

        Args:
            images: One of:
                - [B, 3, H, W]       — single frame per sample
                - [B, T, 3, H, W]    — T frames per sample (video)

        Returns:
            [B, N, output_dim] or [B, T*N, output_dim] for multi-frame inputs.
        """
        if images.ndim == 5:
            # Multi-frame: flatten B and T, process independently, then restore
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)
            tokens_flat = self._encode_single(images_flat)   # [B*T, N, D]
            N, D = tokens_flat.shape[1], tokens_flat.shape[2]
            tokens = tokens_flat.view(B, T * N, D)           # [B, T*N, D]
        else:
            tokens = self._encode_single(images)             # [B, N, D]

        if self.proj is not None:
            tokens = self.proj(tokens)
        return tokens

    def _encode_single(self, images: Tensor) -> Tensor:
        """Encode a batch of single-frame images.

        Args:
            images: [B, 3, H, W]

        Returns:
            [B, N, D] where N = number of tokens and D = backbone embed_dim.
        """
        if self._freeze_encoder:
            with torch.no_grad():
                out = self.backbone(images)
        else:
            out = self.backbone(images)

        # ViT / DINOv2 / EVA-02 / DeiT-III: patch tokens available directly
        if self.return_patch_tokens and out.patch_tokens is not None:
            return out.patch_tokens                          # [B, N, D]

        # CNN backbones: use finest feature map as token grid
        if out.feature_maps:
            feat = out.feature_maps[0]                      # [B, C, H', W']
            B, C, H, W = feat.shape
            return feat.flatten(2).transpose(1, 2)          # [B, H'*W', C]

        # Fallback: single pooled token
        return out.pooled_output.unsqueeze(1)                # [B, 1, D]

    def __repr__(self) -> str:
        return (
            f"HaloVisionBackbone("
            f"backbone={self.backbone_name}, "
            f"output_dim={self.output_dim}, "
            f"frozen={self._freeze_encoder})"
        )


def build_halo_vision_backbone(
    backbone_name:       str          = "convnext_base",
    img_size:            int          = 224,
    freeze_encoder:      bool         = True,
    freeze_up_to_stage:  int          = -1,
    pretrained_path:     Optional[str] = None,
    return_patch_tokens: bool          = True,
    proj_dim:            Optional[int] = None,
) -> HaloVisionBackbone:
    """Factory function for HaloVisionBackbone.

    Example::

        from Halo_VLA.models.vision_backbone import build_halo_vision_backbone

        backbone = build_halo_vision_backbone(
            "dino_vit_giant",
            img_size=224,
            freeze_encoder=True,    # use DINOv2 as frozen feature extractor
            proj_dim=2048,          # project 1536-dim DINOv2 to 2048-dim LLM space
        )
        tokens = backbone(images)   # [B, N, 2048]
    """
    return HaloVisionBackbone(
        backbone_name=backbone_name,
        img_size=img_size,
        freeze_encoder=freeze_encoder,
        freeze_up_to_stage=freeze_up_to_stage,
        pretrained_path=pretrained_path,
        return_patch_tokens=return_patch_tokens,
        proj_dim=proj_dim,
    )
