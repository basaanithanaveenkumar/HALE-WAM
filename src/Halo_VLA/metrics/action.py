"""
Action prediction metrics for HALE-WAM.

Robot action evaluation differs from classification/detection:
  - Actions are continuous vectors in joint/task space (e.g. 7-DoF arm: [Δq1..Δq7]).
  - Temporal structure matters — a prediction that diverges early is worse than
    one that diverges late, because the robot executes the whole chunk sequentially.
  - Action dimensions have very different ranges (e.g. wrist rotation −π..π vs
    finger force 0..0.1 N), so raw L1 must be normalised to compare fairly.

All metrics follow the ``reset / update / compute`` lifecycle from ``base.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .base import RoboticsMetricBase


# ─── helpers ────────────────────────────────────────────────────────────────

def _apply_mask(t: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Flatten a [B, T, D] tensor, applying a [B, T] boolean mask if given."""
    if mask is None:
        return t.reshape(-1, t.shape[-1])
    # mask: True where the timestep is valid
    return t[mask.bool()]  # [N_valid, D]


# ─── ActionL1Metric ──────────────────────────────────────────────────────────

class ActionL1Metric(RoboticsMetricBase):
    """Mean absolute error between predicted and target action sequences.

    WHY L1 over L2:
        Robot action errors from grasping failures or collisions tend to be
        large and sparse — outliers in the MSE sense. L1 is more robust to
        these tail events and gives a better picture of *typical* error.

    Per-dimension breakdown reveals which DoFs are hardest (e.g. wrist yaw
    is typically harder than elbow pitch).  Per-timestep breakdown shows
    where within the chunk the model loses accuracy (late-chunk errors are
    expected; early-chunk errors indicate a fundamental prediction failure).

    Args:
        action_dim: number of action dimensions (e.g. 7 for a 7-DoF arm).
    """

    name = "action_l1"

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        # Total absolute error and count — used for the overall mean
        self._total_err = torch.zeros(self.action_dim)   # [D]
        self._n = 0

        # Per-timestep accumulators (kept for up to 3 timesteps)
        self._t_err: list[torch.Tensor] = [torch.zeros(self.action_dim) for _ in range(3)]
        self._t_n: list[int] = [0, 0, 0]

    def update(
        self,
        pred: Tensor,           # [B, T, action_dim]
        target: Tensor,         # [B, T, action_dim]
        mask: Optional[Tensor] = None,  # [B, T] — True where timestep is valid
    ) -> None:
        err = (pred - target).abs().detach().cpu()  # [B, T, D]

        # ── overall ──────────────────────────────────────────────────────────
        flat_err = _apply_mask(err, mask)           # [N_valid, D]
        self._total_err += flat_err.sum(0)
        self._n += flat_err.shape[0]

        # ── per-timestep (first 3 steps) ─────────────────────────────────────
        T = err.shape[1]
        for ti in range(min(T, 3)):
            if mask is not None:
                step_mask = mask[:, ti].bool()
                step_err = err[:, ti][step_mask]    # [N_valid_at_t, D]
            else:
                step_err = err[:, ti]               # [B, D]
            self._t_err[ti] += step_err.sum(0)
            self._t_n[ti] += step_err.shape[0]

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        mean_per_dim = self._total_err / n           # [D]

        out: dict[str, float] = {
            # Mean L1 over all timesteps and all dimensions
            "action_l1": float(mean_per_dim.mean()),
        }

        # Per-dimension breakdown (dim_0 … dim_{D-1})
        for d, v in enumerate(mean_per_dim.tolist()):
            out[f"action_l1_dim_{d}"] = v

        # Per-timestep (first 3 steps)
        for ti in range(3):
            n_t = max(self._t_n[ti], 1)
            out[f"action_l1_t{ti + 1}"] = float((self._t_err[ti] / n_t).mean())

        return out


# ─── ActionMSEMetric ─────────────────────────────────────────────────────────

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
        self._total_sq = torch.zeros(self.action_dim)
        self._n = 0

    def update(
        self,
        pred: Tensor,                   # [B, T, action_dim]
        target: Tensor,                 # [B, T, action_dim]
        mask: Optional[Tensor] = None,  # [B, T]
    ) -> None:
        sq_err = ((pred - target) ** 2).detach().cpu()  # [B, T, D]
        flat = _apply_mask(sq_err, mask)                 # [N_valid, D]
        self._total_sq += flat.sum(0)
        self._n += flat.shape[0]

    def compute(self) -> dict[str, float]:
        n = max(self._n, 1)
        mean_sq = self._total_sq / n        # [D]
        mse = float(mean_sq.mean())
        rmse = math.sqrt(mse)
        return {
            "action_mse": mse,
            "action_rmse": rmse,
        }


# ─── ActionChunkAccuracyMetric ───────────────────────────────────────────────

class ActionChunkAccuracyMetric(RoboticsMetricBase):
    """Fraction of action chunks with maximum per-step L1 error below a threshold.

    WHY chunk-level accuracy:
        A chunk is the full action sequence the robot commits to executing.
        If *any* step in the chunk exceeds the threshold, the robot will
        deviate from the plan — so it is natural to evaluate the chunk
        atomically rather than averaging per-step errors.

        Three thresholds capture:
          0.05  — tight tolerance (precision tasks: peg insertion, screw driving)
          0.10  — medium tolerance (pick-and-place, folding)
          0.20  — loose tolerance (navigation, coarse manipulation)

    Args:
        thresholds: L1 thresholds at which to report accuracy.
    """

    name = "chunk_acc"

    def __init__(self, thresholds: tuple[float, ...] = (0.05, 0.1, 0.2)) -> None:
        self.thresholds = thresholds
        self.reset()

    def reset(self) -> None:
        self._correct = [0] * len(self.thresholds)
        self._total = 0

    def update(
        self,
        pred: Tensor,    # [B, T, action_dim]
        target: Tensor,  # [B, T, action_dim]
    ) -> None:
        # max L1 error per sample across all timesteps and dimensions
        max_err = (pred - target).abs().max(dim=-1).values.max(dim=-1).values  # [B]
        max_err = max_err.detach().cpu()
        self._total += max_err.shape[0]
        for i, thr in enumerate(self.thresholds):
            self._correct[i] += int((max_err < thr).sum().item())

    def compute(self) -> dict[str, float]:
        total = max(self._total, 1)
        out: dict[str, float] = {}
        for i, thr in enumerate(self.thresholds):
            key = f"chunk_acc_{thr:.2f}".replace("0.", "0.")
            out[key] = self._correct[i] / total
        return out


# ─── TrajectorySmoothnessMetric ──────────────────────────────────────────────

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


# ─── ActionNormalisedError ───────────────────────────────────────────────────

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
