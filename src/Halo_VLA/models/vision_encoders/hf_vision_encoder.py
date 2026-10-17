"""HFVisionEncoder: hFVisionEncoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from exceptions import HaloVLAConfigError

#: Statistics the VLA datasets normalise images with (``img_mean`` / ``img_std``).
DATASET_MEAN = (0.485, 0.456, 0.406)


DATASET_STD = (0.229, 0.224, 0.225)


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
            raise HaloVLAConfigError(f"{model_id} expects {native}px images but img_size={img_size}; set img_size={native}")
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
