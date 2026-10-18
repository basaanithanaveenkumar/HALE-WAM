"""EpisodeLengthMetric: episode length and efficiency — how many steps does the policy use?."""

from __future__ import annotations

import torch

from ..base import RoboticsMetricBase


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
        """Forget the recorded episode lengths."""
        self._steps: list[int] = []

    def update(self, episode_steps: int, max_steps: int | None = None) -> None:  # type: ignore[override]
        """
        Args:
            episode_steps: steps taken in the episode.
            max_steps:     per-episode override for the timeout threshold.
        """
        self._steps.append(episode_steps)

    def compute(self) -> dict[str, float]:
        """Return the metrics.

        Returns:
            ``mean_episode_steps``, ``efficiency`` (1 - steps / budget) and ``timeout_rate``.
        """
        if not self._steps:
            return {
                "mean_episode_steps": 0.0,
                "efficiency": 0.0,
                "timeout_rate": 0.0,
            }
        s = torch.tensor(self._steps, dtype=torch.float32)
        mean_steps = float(s.mean().item())
        efficiency = 1.0 - mean_steps / self.max_steps  # 0 = used full budget; 1 = instant
        timeout_rate = float((s >= self.max_steps).float().mean().item())
        return {
            "mean_episode_steps": mean_steps,
            "efficiency": max(efficiency, 0.0),  # clamp: >max_steps possible
            "timeout_rate": timeout_rate,
        }
