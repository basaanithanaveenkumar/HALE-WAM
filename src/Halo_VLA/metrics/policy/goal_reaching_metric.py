"""GoalReachingMetric: euclidean distance between the final end-effector position and the goal."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ..base import RoboticsMetricBase


class GoalReachingMetric(RoboticsMetricBase):
    """Euclidean distance between the final end-effector position and the goal.

    WHY end-effector distance:
        Many manipulation tasks ultimately require placing the end-effector (or
        a grasped object) within a spatial tolerance of a target.  This metric
        measures that directly, separate from the action sequence quality.

        Thresholded accuracy (e.g. < 5 cm) is the standard evaluation used in
        many manipulation benchmarks (BC-Z, RoboAgent, etc.).

    Args:
        thresholds_m: distances (in metres) below which the episode counts as
                      a "reach success".  Defaults to 5 cm and 10 cm.
    """

    name = "goal_reaching"

    def __init__(self, thresholds_m: tuple[float, ...] = (0.05, 0.10)) -> None:
        self.thresholds_m = thresholds_m
        self.reset()

    def reset(self) -> None:
        """Forget the recorded distances."""
        self._distances: List[float] = []

    def update(
        self,
        final_pos: Tensor,  # [3] — final end-effector XYZ in world frame (metres)
        goal_pos: Tensor,   # [3] — goal XYZ in world frame (metres)
    ) -> None:  # type: ignore[override]
        """Record one episode.

        Args:
            final_pos: final end-effector position ``[3]`` in metres.
            goal_pos: goal position ``[3]`` in metres.
        """
        dist = float((final_pos.float() - goal_pos.float()).norm(p=2).item())
        self._distances.append(dist)

    def compute(self) -> dict[str, float]:
        """Return the metrics.

        Returns:
            ``goal_dist_mean``, ``goal_dist_std`` and ``goal_reach_rate_<threshold>`` per threshold.
        """
        if not self._distances:
            base = {"goal_dist_mean": 0.0, "goal_dist_std": 0.0}
            for t in self.thresholds_m:
                base[f"goal_reach_rate_{t:.2f}"] = 0.0
            return base

        d = torch.tensor(self._distances)
        out: dict[str, float] = {
            "goal_dist_mean": float(d.mean().item()),
            "goal_dist_std":  float(d.std(correction=0).item()),
        }
        for t in self.thresholds_m:
            rate = float((d < t).float().mean().item())
            out[f"goal_reach_rate_{t:.2f}"] = rate
        return out
