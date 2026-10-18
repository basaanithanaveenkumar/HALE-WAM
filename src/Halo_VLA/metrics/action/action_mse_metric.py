"""ActionMSEMetric: mean squared error and RMSE between predicted and target action sequences."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from ..base import RoboticsMetricBase
from .common import _apply_mask


class ActionMSEMetric(RoboticsMetricBase):
    """Mean squared error and RMSE between predicted and target action sequences.

    WHY MSE alongside L1:
        MSE penalises large errors quadratically, making it sensitive to
        safety-critical mistakes (e.g. a 30° joint-limit overshoot matters
        far more than its linear magnitude suggests).  Track both L1 and MSE:
        - If MSE >> L1²:  occasional large errors (dangerous outliers).
        - If MSE ≈ L1²:  errors are roughly uniform (normal regression noise).

    Args:
        action_dim: number of action dimensions.
    """

    name = "action_mse"

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated squared-error sums."""
        self._total_sq = torch.zeros(self.action_dim)
        self._n = 0

    def update(
        self,
        pred: Tensor,                   # [B, T, action_dim]
        target: Tensor,                 # [B, T, action_dim]
        mask: Optional[Tensor] = None,  # [B, T]
    ) -> None:
        """Accumulate one batch.

        Args:
            pred: predicted actions ``[B, T, action_dim]``.
            target: ground-truth actions ``[B, T, action_dim]``.
            mask: ``[B, T]``, True where the timestep is valid (default: all valid).
        """
        sq_err = ((pred - target) ** 2).detach().cpu()  # [B, T, D]
        flat = _apply_mask(sq_err, mask)                 # [N_valid, D]
        self._total_sq += flat.sum(0)
        self._n += flat.shape[0]

    def compute(self) -> dict[str, float]:
        """Return the metrics.

        Returns:
            ``action_mse`` and ``action_rmse``.
        """
        n = max(self._n, 1)
        mean_sq = self._total_sq / n        # [D]
        mse = float(mean_sq.mean())
        rmse = math.sqrt(mse)
        return {
            "action_mse": mse,
            "action_rmse": rmse,
        }
