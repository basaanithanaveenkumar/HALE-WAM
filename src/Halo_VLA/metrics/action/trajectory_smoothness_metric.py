"""TrajectorySmoothnessMetric: mean squared second-order finite difference (jerk proxy) of trajectories."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..base import RoboticsMetricBase


class TrajectorySmoothnessMetric(RoboticsMetricBase):
    """Mean squared second-order finite difference (jerk proxy) of trajectories.

    Jerk in robotics = rate of change of acceleration.  For discrete sequences
    we use the second-order difference as a jerk proxy:

        jerk[t] = a[t+2] − 2·a[t+1] + a[t]

    WHY smoothness matters:
        Jerky trajectories cause mechanical wear, unsafe oscillations, and
        difficulty for downstream robot controllers to track.  A low smoothness
        score indicates the model learned physically plausible motions.

    Lower = smoother.  A perfectly linear trajectory scores 0.
    """

    name = "smoothness"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._pred_jerk_sum = 0.0
        self._tgt_jerk_sum = 0.0
        self._n = 0
        self._has_target = False

    def update(
        self,
        pred: Tensor,                   # [B, T, action_dim]
        target: Optional[Tensor] = None,  # [B, T, action_dim]
    ) -> None:
        if pred.shape[1] < 3:
            return  # need at least 3 timesteps for second-order diff

        # Second-order finite difference: Δ²a[t] = a[t+2] - 2*a[t+1] + a[t]
        jerk_pred = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # [B, T-2, D]
        self._pred_jerk_sum += float((jerk_pred ** 2).mean().item()) * pred.shape[0]
        self._n += pred.shape[0]

        if target is not None:
            jerk_tgt = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
            self._tgt_jerk_sum += float((jerk_tgt ** 2).mean().item()) * target.shape[0]
            self._has_target = True

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        out = {"pred_smoothness": self._pred_jerk_sum / n}
        if self._has_target:
            out["target_smoothness"] = self._tgt_jerk_sum / n
        return out
