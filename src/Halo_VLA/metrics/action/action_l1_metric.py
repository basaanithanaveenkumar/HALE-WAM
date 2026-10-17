"""ActionL1Metric: mean absolute error between predicted and target action sequences."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..base import RoboticsMetricBase
from .common import _apply_mask


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
