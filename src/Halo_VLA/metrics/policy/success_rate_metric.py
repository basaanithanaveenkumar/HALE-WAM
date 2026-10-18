"""SuccessRateMetric: binary task success rate over a set of evaluation episodes."""

from __future__ import annotations

from ..base import RoboticsMetricBase


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
        """Clear the success counters."""
        self._n_success = 0
        self._n_total = 0

    def update(self, success: bool) -> None:  # type: ignore[override]
        """Record one episode.

        Args:
            success: whether the episode succeeded.
        """
        self._n_total += 1
        if success:
            self._n_success += 1

    def compute(self) -> dict[str, float]:
        """Return the metrics.

        Returns:
            ``success_rate``, ``n_episodes`` and ``n_success``.
        """
        return {
            "success_rate": self._n_success / max(self._n_total, 1),
            "n_episodes": float(self._n_total),
            "n_success": float(self._n_success),
        }
