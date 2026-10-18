"""FVDProxyMetric: lightweight proxy for Fréchet Video Distance (FVD)."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ..base import RoboticsMetricBase


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
        """Clear the collected per-frame feature vectors."""
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
        """Collect per-frame statistics of one batch of videos.

        Args:
            pred_frames: predicted videos ``[B, T, C, H, W]``.
            target_frames: ground-truth videos ``[B, T, C, H, W]``.
        """
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
        """Return the metrics.

        Returns:
            ``pixel_fvd_proxy``: the Fréchet distance between predicted and target feature sets
            (0 when nothing was collected).
        """
        if not self._pred_feats:
            return {"pixel_fvd_proxy": 0.0}
        pred_all = torch.cat(self._pred_feats, dim=0)   # [N_total, 2C]
        tgt_all  = torch.cat(self._tgt_feats,  dim=0)   # [N_total, 2C]
        fd = self._frechet_distance(pred_all, tgt_all)
        return {"pixel_fvd_proxy": max(fd, 0.0)}  # clamp: numerical issues can give tiny negatives
