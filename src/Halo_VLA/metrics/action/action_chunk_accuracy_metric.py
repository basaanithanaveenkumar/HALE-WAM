"""ActionChunkAccuracyMetric: fraction of action chunks with maximum per-step L1 error below a threshold."""

from __future__ import annotations

from torch import Tensor

from ..base import RoboticsMetricBase


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
