"""ActionNormalisedError: l1 error normalised by the observed range of each action dimension."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..base import RoboticsMetricBase
from .common import _apply_mask


class ActionNormalisedError(RoboticsMetricBase):
    """L1 error normalised by the observed range of each action dimension.

    WHY normalise by range:
        A 7-DoF robot arm might have joint 1 spanning [−π, π] and joint 6
        (finger) spanning [0, 0.1].  A raw L1 of 0.01 rad means very different
        things for these two joints.  Normalising by the observed range gives
        each dimension equal weight and makes cross-task comparisons fair.

    The min/max of each dimension is estimated from the *target* actions seen
    during ``update()`` calls (not the predictions), which gives the true task
    range independent of model quality.

    Args:
        action_dim: number of action dimensions.
        eps: small constant to avoid division by zero for constant dimensions.
    """

    name = "normalised_action_error"

    def __init__(self, action_dim: int, eps: float = 1e-6) -> None:
        self.action_dim = action_dim
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self._dim_min = torch.full((self.action_dim,), float("inf"))
        self._dim_max = torch.full((self.action_dim,), float("-inf"))
        self._total_err = torch.zeros(self.action_dim)
        self._n = 0

    def update(
        self,
        pred: Tensor,                   # [B, T, action_dim]
        target: Tensor,                 # [B, T, action_dim]
        mask: Optional[Tensor] = None,  # [B, T]
    ) -> None:
        pred_f = _apply_mask(pred.detach().cpu(), mask)    # [N, D]
        tgt_f  = _apply_mask(target.detach().cpu(), mask)  # [N, D]

        # Track observed range from target (ground truth)
        self._dim_min = torch.minimum(self._dim_min, tgt_f.min(0).values)
        self._dim_max = torch.maximum(self._dim_max, tgt_f.max(0).values)

        self._total_err += (pred_f - tgt_f).abs().sum(0)
        self._n += pred_f.shape[0]

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        mean_err = self._total_err / n                            # [D]
        dim_range = (self._dim_max - self._dim_min).clamp(min=self.eps)  # [D]
        normalised = mean_err / dim_range                         # [D]
        return {"normalised_action_error": float(normalised.mean())}
