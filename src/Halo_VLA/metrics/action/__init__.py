"""Action prediction metrics for HALE-WAM.

Robot action evaluation differs from classification/detection:
  - Actions are continuous vectors in joint/task space (e.g. 7-DoF arm: [Δq1..Δq7]).
  - Temporal structure matters — a prediction that diverges early is worse than
    one that diverges late, because the robot executes the whole chunk sequentially.
  - Action dimensions have very different ranges (e.g. wrist rotation −π..π vs
    finger force 0..0.1 N), so raw L1 must be normalised to compare fairly.

All metrics follow the ``reset / update / compute`` lifecycle from ``base.py``."""

from .action_chunk_accuracy_metric import ActionChunkAccuracyMetric  # noqa: F401
from .action_l1_metric import ActionL1Metric  # noqa: F401
from .action_mse_metric import ActionMSEMetric  # noqa: F401
from .action_normalised_error import ActionNormalisedError  # noqa: F401
from .common import _apply_mask  # noqa: F401
from .trajectory_smoothness_metric import TrajectorySmoothnessMetric  # noqa: F401

__all__ = [
    "ActionChunkAccuracyMetric",
    "ActionL1Metric",
    "ActionMSEMetric",
    "ActionNormalisedError",
    "TrajectorySmoothnessMetric",
]
