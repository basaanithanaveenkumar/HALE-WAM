"""VAEOutput: all outputs from a VideoVAE forward pass (used during joint training)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VAEOutput:
    """All outputs from a VideoVAE forward pass (used during joint training)."""
    x_recon: torch.Tensor   # [B, 3, H, W]        pixel reconstruction
    z:       torch.Tensor   # [B, 4, H/8, W/8]    scaled latent (×SCALE_FACTOR)
    mean:    torch.Tensor   # [B, 4, H/8, W/8]    posterior mean
    logvar:  torch.Tensor   # [B, 4, H/8, W/8]    posterior log-variance (clamped)
    kl_loss: torch.Tensor   # scalar               KL( q(z|x) || N(0,I) )
