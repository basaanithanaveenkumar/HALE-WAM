"""SSIMMetric: 1D Gaussian weights → outer-product to get 2D kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import RoboticsMetricBase


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> Tensor:
    """1D Gaussian weights → outer-product to get 2D kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)  # [K, K]


def _ssim_map(
    pred: Tensor,   # [B, C, H, W]
    target: Tensor, # [B, C, H, W]
    kernel: Tensor, # [K, K]
) -> Tensor:
    """Compute per-pixel SSIM map and return the mean over spatial dims."""
    B, C, H, W = pred.shape
    K = kernel.shape[0]
    pad = K // 2

    # Expand kernel to [C, 1, K, K] for depthwise conv
    kk = kernel.view(1, 1, K, K).expand(C, 1, K, K).to(pred.device)

    def smooth(x: Tensor) -> Tensor:
        return F.conv2d(x, kk, padding=pad, groups=C)

    mu1 = smooth(pred)
    mu2 = smooth(target)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = smooth(pred ** 2) - mu1_sq
    sigma2_sq = smooth(target ** 2) - mu2_sq
    sigma12   = smooth(pred * target) - mu1_mu2

    # SSIM constants (Luminance/Contrast stabilisation; Wang et al. 2004)
    C1 = (0.01 * 1.0) ** 2   # k1=0.01, L=1
    C2 = (0.03 * 1.0) ** 2   # k2=0.03

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map    = numerator / denominator.clamp(min=1e-8)
    return ssim_map.mean()


class SSIMMetric(RoboticsMetricBase):
    """Structural Similarity Index between predicted and target frames.

    WHY SSIM over PSNR:
        The human visual system perceives structural patterns rather than
        absolute pixel values.  Two images that look identical to a human
        can have very different PSNR if one is slightly shifted — but
        SSIM will correctly score them as similar.

        For robot video prediction, SSIM captures whether the scene layout
        and object shapes are preserved even if there is mild per-pixel noise.

    Scores range from −1 to 1; 1 = identical. Above 0.95 is visually similar.

    Pure PyTorch implementation using a Gaussian window (no cv2/scikit-image).

    Args:
        kernel_size: Gaussian window side length (default 11, standard value).
        sigma:       Gaussian standard deviation (default 1.5).
    """

    name = "ssim"

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        self._kernel = _gaussian_kernel(kernel_size, sigma)
        self.reset()

    def reset(self) -> None:
        self._total = 0.0
        self._n = 0

    def update(self, pred: Tensor, target: Tensor) -> None:
        s = float(_ssim_map(pred.float(), target.float(), self._kernel).item())
        self._total += s * pred.shape[0]
        self._n += pred.shape[0]

    def compute(self) -> dict[str, float]:
        return {"ssim_mean": self._total / max(self._n, 1)}
