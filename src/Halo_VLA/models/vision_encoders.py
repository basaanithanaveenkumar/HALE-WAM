"""Configurable vision encoder for Halo-VLA: built-in ViT, SigLIP / CLIP, or any HaleBlocks backbone.

Mirrors Hale-VLM's ``vision.encoder: siglip | clip | local`` so both models can use the same
pretrained towers. Every encoder maps ``(B, 3, H, W)`` images (normalised with the datasets'
ImageNet statistics) to ``(B, N, D)`` tokens; the image projector then maps ``D -> emb_dim``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

#: Statistics the VLA datasets normalise images with (``img_mean`` / ``img_std``).
DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)

#: Default Hub repos for the pretrained encoders (both checked on the Hub).
DEFAULT_HF_ENCODERS = {
    "siglip": "google/siglip-base-patch16-224",  # Zhai et al., 2023, arXiv:2303.15343
    "clip": "openai/clip-vit-base-patch16",      # Radford et al., 2021, arXiv:2103.00020
}
ENCODER_TYPES = ("vit", "siglip", "clip", "local")


class HFVisionEncoder(nn.Module):
    """A pretrained SigLIP / CLIP vision tower from the Hub.

    Images arrive normalised with ImageNet statistics; they are re-normalised to the encoder's
    own statistics (read from its image processor) so pretrained features are not distorted.
    When frozen the tower runs without gradients and stays in eval mode even under
    ``model.train()``.

    Args:
        encoder: ``siglip`` or ``clip``.
        model_id: Hub repo or local path.
        img_size: input resolution of the VLA; must equal the encoder's native resolution.
        freeze: keep the tower's weights fixed.
    """

    def __init__(self, encoder: str, model_id: str, img_size: int, freeze: bool) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, CLIPVisionModel, SiglipVisionModel  # lazy

        cls = SiglipVisionModel if encoder == "siglip" else CLIPVisionModel
        self.model = cls.from_pretrained(model_id)
        native = int(self.model.config.image_size)
        if native != img_size:
            raise ValueError(f"{model_id} expects {native}px images but img_size={img_size}; set img_size={native}")
        try:
            proc = AutoImageProcessor.from_pretrained(model_id)
            mean, std = proc.image_mean, proc.image_std
        except (OSError, ValueError):  # no processor config: SigLIP's convention
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        # x_enc = (x_data * std_d + mean_d - mean_e) / std_e  ==  x_data * scale + shift
        scale = torch.tensor(DATASET_STD) / torch.tensor(std)
        shift = (torch.tensor(DATASET_MEAN) - torch.tensor(mean)) / torch.tensor(std)
        self.register_buffer("scale", scale.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", shift.view(1, 3, 1, 1), persistent=False)
        self.frozen = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def train(self, mode: bool = True) -> HFVisionEncoder:
        super().train(mode)
        if self.frozen:
            self.model.eval()  # frozen towers never switch dropout / norms to training mode
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = images * self.scale + self.shift
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            return self.model(pixel_values=pixel_values.to(self.model.dtype)).last_hidden_state.float()


@dataclass(frozen=True)
class VisionEncoderInfo:
    """What the rest of the VLA needs to know about the encoder."""

    output_dim: int
    num_tokens: int


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
        raise ValueError(f"vision_encoder must be one of {ENCODER_TYPES}, got {kind!r}")
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
