"""Losses, config factory and Euler sampler used by the visual DiT frame predictor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from models.DiT import DiT


# ---------------------------------------------------------------------------
# Perceptual loss (VGG feature matching)
# ---------------------------------------------------------------------------
class _VGGPerceptualLoss(nn.Module):
    """Lightweight VGG-based perceptual loss using relu1_2 and relu2_2 features."""

    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as tvm

            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        except TypeError:
            import torchvision.models as tvm

            vgg = tvm.vgg16(pretrained=True)
        # relu1_2 = layers 0-3, relu2_2 = layers 0-8
        features = vgg.features
        self.slice1 = nn.Sequential(*list(features.children())[:4]).eval()
        self.slice2 = nn.Sequential(*list(features.children())[4:9]).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        # ImageNet mean/std for normalisation (VGG expects it)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Perceptual distance between two image batches.

        Args:
            pred: predicted images ``[B, 3, H, W]`` in ``[0, 1]``.
            target: target images, same shape.

        Returns:
            Sum of the feature MSEs at ``relu1_2`` and ``relu2_2``.
        """
        pred_n, tgt_n = self._norm(pred.clamp(0, 1)), self._norm(target.clamp(0, 1))
        f1_p, f1_t = self.slice1(pred_n), self.slice1(tgt_n)
        f2_p, f2_t = self.slice2(f1_p), self.slice2(f1_t)
        return F.mse_loss(f1_p, f1_t) + F.mse_loss(f2_p, f2_t)


# Module-level singleton — built lazily so import doesn't trigger torchvision download.
_vgg_loss: _VGGPerceptualLoss | None = None
_vgg_unavailable = False


def _get_vgg_loss(device: torch.device) -> _VGGPerceptualLoss | None:
    """The shared VGG perceptual-loss network on ``device``, or ``None`` if its weights are unavailable.

    Fetching the ImageNet weights needs network access on first use; when that fails the
    perceptual term is skipped (with one warning) rather than aborting training.
    """
    global _vgg_loss, _vgg_unavailable
    if _vgg_unavailable:
        return None
    if _vgg_loss is None:
        try:
            _vgg_loss = _VGGPerceptualLoss()
        except (OSError, RuntimeError, ValueError) as exc:  # download / checkpoint-loading failures
            _vgg_unavailable = True
            logger.warning(
                "VGG perceptual loss disabled: could not load the ImageNet weights ({})", exc
            )
            return None
    return _vgg_loss.to(device)


# ---------------------------------------------------------------------------
# SSIM loss
# ---------------------------------------------------------------------------
def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 - SSIM, averaged over the batch. pred/target in [0, 1], shape [B, C, H, W]."""
    C1, C2 = 0.01**2, 0.03**2
    # Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]
    C = pred.size(1)
    kernel = kernel.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def mu(x: torch.Tensor) -> torch.Tensor:
        """Gaussian-windowed local mean."""
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_p, mu_t = mu(pred), mu(target)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t
    sigma_p2 = mu(pred**2) - mu_p2
    sigma_t2 = mu(target**2) - mu_t2
    sigma_pt = mu(pred * target) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
        (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)
    )
    return 1.0 - ssim_map.mean()


def make_dit_config(
    base_config: Any,
    in_channels: int,
    out_channels: int | None = None,
) -> SimpleNamespace:
    """
    Build a namespace with configuration attributes for DiT initialization.

    This helper encapsulates common config extraction logic, defaulting to sensible
    values if attributes are missing from the base config.

    Args:
        base_config: Base config object (e.g., from training setup).
        in_channels: Number of input channels for the DiT (e.g., 3 for RGB).
        out_channels: Number of output channels. Defaults to in_channels if None.

    Returns:
        SimpleNamespace with DiT config attributes.
    """
    if out_channels is None:
        out_channels = in_channels
    return SimpleNamespace(
        dit_in_channels=in_channels,
        dit_out_channels=out_channels,
        dit_patch_size=getattr(
            base_config, "dit_patch_size", getattr(base_config, "patch_size", 4)
        ),
        dit_hidden_size=getattr(
            base_config, "dit_hidden_size", getattr(base_config, "emb_dim", 768)
        ),
        dit_depth=getattr(base_config, "dit_depth", 12),
        dit_num_heads=getattr(base_config, "dit_num_heads", 12),
        dit_mlp_ratio=getattr(base_config, "dit_mlp_ratio", 4.0),
        dit_max_resolution=getattr(base_config, "dit_max_resolution", 256),
        emb_dim=base_config.emb_dim,
        dit_time_freq_dim=getattr(base_config, "dit_time_freq_dim", 256),
    )


def flow_matching_velocity_loss(
    dit: DiT,
    x1: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Conditional flow matching loss for a single frame prediction.

    Samples a random time point t ∈ [0, 1] and constructs the OT path:
      x_t = (1 - t) * x_0 + t * x_1
    where x_0 is Gaussian noise and x_1 is the target RGB frame.

    The DiT predicts the velocity v = dx/dt ≈ x_1 - x_0, and we minimize
    the MSE between predicted and actual velocity.

    Args:
        dit: DiT model for velocity prediction.
        x1: [B, C, H, W] target frame (x_1 in the OT path).
        context: [B, emb_dim] conditional embedding from text-action cross-attention.

    Returns:
        Tuple of:
          - loss: scalar MSE loss
          - x_t: [B, C, H, W] noisy sample at time t (useful for auxiliary losses)
          - t: [B] time values
          - v_pred: [B, C, H, W] predicted velocity
    """
    b = x1.size(0)
    device = x1.device

    # Sample random time steps from Uniform[0, 1]
    t = torch.rand(b, device=device)

    # Sample Gaussian noise as x_0
    x0 = torch.randn_like(x1)

    # Construct noisy interpolation: x_t = (1-t) * x_0 + t * x_1
    te = t.view(b, 1, 1, 1)
    x_t = (1.0 - te) * x0 + te * x1

    # Target velocity (ground truth): v* = x_1 - x_0
    v_star = x1 - x0

    # Predict velocity via DiT
    v_pred = dit(x_t, t, context)

    # Compute MSE loss
    loss = F.mse_loss(v_pred, v_star)

    return loss, x_t, t, v_pred


@torch.no_grad()
def sample_di_t_euler(
    dit: DiT,
    context: torch.Tensor,
    channels: int,
    height: int,
    width: int,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Generate an RGB frame by integrating the DiT velocity field using Euler's method.

    Starts from Gaussian noise at t=0 and integrates forward to t=1 using fixed-step
    Euler scheme: x_{i+1} = x_i + v(x_i, t_i) * dt.

    Args:
        dit: DiT model for velocity prediction.
        context: [B, emb_dim] conditional embedding from text-action cross-attention.
        channels: Number of output channels (e.g., 3 for RGB).
        height: Height of the output frame in latent space.
        width: Width of the output frame in latent space.
        num_steps: Number of integration steps (higher = more accurate but slower).

    Returns:
        [B, channels, height, width] generated frame in latent space.
    """
    b = context.size(0)
    device = context.device

    # Initialize with Gaussian noise at t=0
    x = torch.randn(b, channels, height, width, device=device)

    # Compute time step size
    dt = 1.0 / num_steps

    # Euler integration from t=0 to t=1
    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((b,), t_val, device=device)

        # Predict velocity at current step
        v = dit(x, t, context)

        # Euler update step
        x = x + v * dt

    return x


def F_func_temporal(v_frames: torch.Tensor) -> torch.Tensor:  # noqa: N802 - name kept for importers
    """L2 difference between consecutive predicted velocity fields [B, F, C, H, W]."""
    return F.mse_loss(v_frames[:, 1:], v_frames[:, :-1])
