"""TaskCompletionMetric: partial task completion rate — fraction of sub-tasks completed."""

from __future__ import annotations

from torch import Tensor

from ..base import RoboticsMetricBase


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
