"""
Policy-level and task-level evaluation metrics for HALE-WAM.

Unlike action-prediction metrics (which compare predicted vs target action
sequences in an offline/supervised sense), policy metrics evaluate the robot's
*actual behaviour* in simulation or the real world.

They answer: "did the robot achieve the goal?" rather than "was the prediction
numerically close?"

Metrics
-------
SuccessRateMetric     — binary task success/failure.
TaskCompletionMetric  — partial sub-task completion scoring.
GoalReachingMetric    — end-effector distance to goal.
EpisodeLengthMetric   — efficiency measured by steps used.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor

from Halo_VLA.metrics.base import RoboticsMetricBase


# ─── SuccessRateMetric ───────────────────────────────────────────────────────

class SuccessRateMetric(RoboticsMetricBase):
    """Binary task success rate over a set of evaluation episodes.

    WHY this is the primary metric:
        All numerical metrics (L1, MSE, SSIM, …) are proxies; the ultimate
        measure of a robot policy is whether it succeeds at the task.
        Success rate should always be reported alongside regression metrics.

    Usage::

        metric = SuccessRateMetric()
        for episode in eval_episodes:
            success = run_episode(policy, env)
            metric.update(success=success)
        print(metric.compute())
    """

    name = "success"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._n_success = 0
        self._n_total = 0

    def update(self, success: bool) -> None:  # type: ignore[override]
        self._n_total += 1
        if success:
            self._n_success += 1

    def compute(self) -> dict[str, float]:
        return {
            "success_rate": self._n_success / max(self._n_total, 1),
            "n_episodes":   float(self._n_total),
            "n_success":    float(self._n_success),
        }


# ─── TaskCompletionMetric ────────────────────────────────────────────────────

class TaskCompletionMetric(RoboticsMetricBase):
    """Partial task completion rate — fraction of sub-tasks completed.

    WHY sub-task completion:
        Binary success/failure discards information when tasks have multiple
        stages (e.g. "open drawer, pick object, place object in drawer, close
        drawer").  A policy that completes 3/4 sub-tasks is better than one
        completing 0/4, even if neither scores a binary success.

        Partial completion provides a richer training signal and is used in
        benchmarks like Language-Table and RLBench.

    Args:
        num_subtasks: fixed number of sub-tasks (set to None for dynamic).
    """

    name = "task_completion"

    def __init__(self, num_subtasks: int | None = None) -> None:
        self.num_subtasks = num_subtasks
        self.reset()

    def reset(self) -> None:
        self._total_completion = 0.0
        self._total_subtasks_done = 0.0
        self._n_episodes = 0

    def update(self, subtask_mask: Tensor) -> None:  # type: ignore[override]
        """
        Args:
            subtask_mask: [K] boolean/float tensor — 1 if sub-task k completed.
        """
        k = subtask_mask.numel()
        done = float(subtask_mask.float().sum().item())
        self._total_subtasks_done += done
        self._total_completion += done / max(k, 1)
        self._n_episodes += 1

    def compute(self) -> dict[str, float]:
        n = max(self._n_episodes, 1)
        return {
            "task_completion_rate": self._total_completion / n,
            "mean_subtasks_done":   self._total_subtasks_done / n,
        }


# ─── GoalReachingMetric ──────────────────────────────────────────────────────

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
        self._distances: List[float] = []

    def update(
        self,
        final_pos: Tensor,  # [3] — final end-effector XYZ in world frame (metres)
        goal_pos: Tensor,   # [3] — goal XYZ in world frame (metres)
    ) -> None:  # type: ignore[override]
        dist = float((final_pos.float() - goal_pos.float()).norm(p=2).item())
        self._distances.append(dist)

    def compute(self) -> dict[str, float]:
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


# ─── EpisodeLengthMetric ─────────────────────────────────────────────────────

class EpisodeLengthMetric(RoboticsMetricBase):
    """Episode length and efficiency — how many steps does the policy use?

    WHY track episode length:
        A policy that succeeds on step 200/200 (timeout) is less capable than
        one succeeding on step 40/200.  Efficiency captures this: it rewards
        completing tasks quickly and penalises spending the full budget.

        Timeout rate is a useful diagnostic — a high rate indicates the policy
        sometimes gets "stuck" (oscillates without making progress).

    Args:
        max_steps: episode step limit (for normalisation and timeout detection).
    """

    name = "episode_length"

    def __init__(self, max_steps: int = 200) -> None:
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> None:
        self._steps: List[int] = []

    def update(self, episode_steps: int, max_steps: int | None = None) -> None:  # type: ignore[override]
        """
        Args:
            episode_steps: steps taken in the episode.
            max_steps:     per-episode override for the timeout threshold.
        """
        self._steps.append(episode_steps)

    def compute(self) -> dict[str, float]:
        if not self._steps:
            return {
                "mean_episode_steps": 0.0,
                "efficiency":         0.0,
                "timeout_rate":       0.0,
            }
        s = torch.tensor(self._steps, dtype=torch.float32)
        mean_steps = float(s.mean().item())
        efficiency = 1.0 - mean_steps / self.max_steps        # 0 = used full budget; 1 = instant
        timeout_rate = float((s >= self.max_steps).float().mean().item())
        return {
            "mean_episode_steps": mean_steps,
            "efficiency":         max(efficiency, 0.0),        # clamp: >max_steps possible
            "timeout_rate":       timeout_rate,
        }
