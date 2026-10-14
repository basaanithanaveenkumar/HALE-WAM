"""
Video and image generation quality metrics for HALE-WAM.

HALE-WAM uses a DiT-based visual predictor (``dit_frame_prediction.py``) to
generate future frames from robot observations.  These metrics evaluate the
quality of that generation.

Metrics
-------
PSNRMetric    — pixel-level fidelity (in dB); quick and deterministic.
SSIMMetric    — structural/perceptual similarity; better than PSNR for blur.
FVDProxyMetric— lightweight proxy for Fréchet Video Distance; captures
               temporal distribution mismatch without the I3D network.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import RoboticsMetricBase


# ─── PSNRMetric ───────────────────────────────────────────────────────────────

class PSNRMetric(RoboticsMetricBase):
    """Peak Signal-to-Noise Ratio between predicted and target image frames.

    PSNR = 10 · log₁₀(MAX² / MSE)

    For float images in [0, 1] MAX = 1.0, so: PSNR = −10 · log₁₀(MSE).

    WHY PSNR:
        PSNR is the industry-standard measure of pixel-level fidelity.
        Values above 30 dB are perceptually similar; above 40 dB is nearly
        indistinguishable.  It is fast (just MSE) and deterministic, making it
        a reliable first-pass metric during training.

    Limitation: PSNR treats all pixels equally — it cannot distinguish a
    blurry image from one with displaced sharp edges.  Use alongside SSIM.

    Args:
        max_val: maximum pixel value (1.0 for float [0,1], 255 for uint8).
    """

    name = "psnr"

    def __init__(self, max_val: float = 1.0) -> None:
        self.max_val = max_val
        self.reset()

    def reset(self) -> None:
        self._total_mse = 0.0
        self._n = 0

    def update(
        self,
        pred: Tensor,    # [B, C, H, W] float32, same range as max_val
        target: Tensor,  # [B, C, H, W]
    ) -> None:
        mse = float(F.mse_loss(pred.float(), target.float()).item())
        self._total_mse += mse * pred.shape[0]
        self._n += pred.shape[0]

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        mean_mse = self._total_mse / n
        if mean_mse < 1e-10:
            psnr = 100.0  # effectively infinite (perfect reconstruction)
        else:
            psnr = 10.0 * math.log10((self.max_val ** 2) / mean_mse)
        return {"psnr_mean": psnr}


# ─── SSIMMetric ───────────────────────────────────────────────────────────────

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


# ─── FVDProxyMetric ───────────────────────────────────────────────────────────

class FVDProxyMetric(RoboticsMetricBase):
    """Lightweight proxy for Fréchet Video Distance (FVD).

    True FVD requires an I3D network pretrained on Kinetics, which is a large
    external dependency.  This proxy computes the Fréchet distance between
    distributions of per-frame pixel statistics (mean and std per channel),
    capturing temporal distribution mismatch without any pretrained model.

    WHY FVD (and this proxy):
        PSNR and SSIM are *paired* metrics — they compare a specific predicted
        frame to a specific target frame.  FVD is *distributional*: it compares
        the statistical distribution of generated videos to the real one.  A
        model that always produces plausible but wrong videos will score well
        on PSNR/SSIM for some samples but poorly on FVD.

        This proxy is useful for quickly detecting distributional collapse
        (e.g. mode-dropping, color distribution shift) during training.

    The proxy Fréchet distance is computed as:
        FD = ||μ_pred − μ_tgt||² + Tr(Σ_pred + Σ_tgt − 2·(Σ_pred·Σ_tgt)^½)
    using the feature vectors of per-frame pixel stats.
    """

    name = "fvd_proxy"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Accumulate (mean, std) feature vectors per frame across all update calls
        self._pred_feats: List[Tensor] = []
        self._tgt_feats: List[Tensor] = []

    def _frame_features(self, frames: Tensor) -> Tensor:
        """Extract per-frame feature vector: [channel_mean × C, channel_std × C].

        Args:
            frames: [B, T, C, H, W]
        Returns:
            feats: [B*T, 2*C]
        """
        B, T, C, H, W = frames.shape
        f = frames.reshape(B * T, C, H, W)
        # Per-channel mean and std over spatial dimensions
        mu  = f.mean(dim=(-2, -1))              # [B*T, C]
        std = f.std(dim=(-2, -1), correction=0) # [B*T, C]
        return torch.cat([mu, std], dim=1)       # [B*T, 2C]

    def update(
        self,
        pred_frames: Tensor,    # [B, T, C, H, W]
        target_frames: Tensor,  # [B, T, C, H, W]
    ) -> None:
        self._pred_feats.append(self._frame_features(pred_frames.float().detach().cpu()))
        self._tgt_feats.append(self._frame_features(target_frames.float().detach().cpu()))

    @staticmethod
    def _frechet_distance(x: Tensor, y: Tensor) -> float:
        """Fréchet distance between two sets of feature vectors.

        Args:
            x, y: [N, D] feature matrices
        Returns:
            Scalar Fréchet distance.
        """
        mu_x = x.mean(0)          # [D]
        mu_y = y.mean(0)          # [D]
        # Covariances
        xc = x - mu_x
        yc = y - mu_y
        sig_x = (xc.T @ xc) / max(x.shape[0] - 1, 1)  # [D, D]
        sig_y = (yc.T @ yc) / max(y.shape[0] - 1, 1)   # [D, D]

        # Matrix square root via eigendecomposition
        # √(Σ_x · Σ_y) ≈ √(Σ_x) · √(Σ_y)  (approximate, but stable)
        mean_diff_sq = float(((mu_x - mu_y) ** 2).sum().item())
        cov_trace = float((sig_x + sig_y).diagonal().sum().item())

        # Approximation of Tr(sqrt(sig_x @ sig_y)) using element-wise sqrt
        # (exact only when matrices commute; acceptable for a proxy metric)
        cov_sqrt_trace = float(((sig_x * sig_y).clamp(min=0).sqrt()).diagonal().sum().item())

        return mean_diff_sq + cov_trace - 2.0 * cov_sqrt_trace

    def compute(self) -> dict[str, float]:
        if not self._pred_feats:
            return {"pixel_fvd_proxy": 0.0}
        pred_all = torch.cat(self._pred_feats, dim=0)   # [N_total, 2C]
        tgt_all  = torch.cat(self._tgt_feats,  dim=0)   # [N_total, 2C]
        fd = self._frechet_distance(pred_all, tgt_all)
        return {"pixel_fvd_proxy": max(fd, 0.0)}  # clamp: numerical issues can give tiny negatives
