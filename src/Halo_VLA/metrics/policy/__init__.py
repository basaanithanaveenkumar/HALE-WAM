"""Policy-level and task-level evaluation metrics for HALE-WAM.

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
EpisodeLengthMetric   — efficiency measured by steps used."""

from .episode_length_metric import EpisodeLengthMetric  # noqa: F401
from .goal_reaching_metric import GoalReachingMetric  # noqa: F401
from .success_rate_metric import SuccessRateMetric  # noqa: F401
from .task_completion_metric import TaskCompletionMetric  # noqa: F401

__all__ = [
    "EpisodeLengthMetric",
    "GoalReachingMetric",
    "SuccessRateMetric",
    "TaskCompletionMetric",
]
