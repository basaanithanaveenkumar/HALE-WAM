"""PSNRMetric: peak Signal-to-Noise Ratio between predicted and target image frames."""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor

from ..base import RoboticsMetricBase


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
        """Clear the accumulated mean squared error."""
        self._total_mse = 0.0
        self._n = 0

    def update(
        self,
        pred: Tensor,    # [B, C, H, W] float32, same range as max_val
        target: Tensor,  # [B, C, H, W]
    ) -> None:
        """Accumulate one batch.

        Args:
            pred: predicted images ``[B, C, H, W]`` in the range of ``max_val``.
            target: ground-truth images ``[B, C, H, W]``.
        """
        mse = float(F.mse_loss(pred.float(), target.float()).item())
        self._total_mse += mse * pred.shape[0]
        self._n += pred.shape[0]

    def compute(self) -> dict[str, float]:
        """Return the metrics.

        Returns:
            ``psnr_mean`` in dB (capped for identical images).
        """
        n = max(self._n, 1)
        mean_mse = self._total_mse / n
        if mean_mse < 1e-10:
            psnr = 100.0  # effectively infinite (perfect reconstruction)
        else:
            psnr = 10.0 * math.log10((self.max_val ** 2) / mean_mse)
        return {"psnr_mean": psnr}
