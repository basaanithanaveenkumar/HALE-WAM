"""
HALE-WAM Evaluation Metrics
============================

Comprehensive metrics for evaluating the Halo-VLA robot foundation model.
All metrics follow a stateful ``reset / update / compute`` lifecycle.

Action Prediction (offline, supervised evaluation)
---------------------------------------------------
ActionL1Metric           — per-step L1 with per-dim and per-timestep breakdown
ActionMSEMetric          — per-step MSE and RMSE
ActionChunkAccuracyMetric — chunk-level accuracy at multiple L1 thresholds
TrajectorySmoothnessMetric— jerk-proxy for trajectory quality
ActionNormalisedError    — L1 normalised by per-dimension observed range

Video Generation Quality
------------------------
PSNRMetric       — pixel-level fidelity (dB); >30 dB = perceptually similar
SSIMMetric       — structural similarity; better than PSNR for blurring artefacts
FVDProxyMetric   — distributional video quality proxy without I3D dependency

Policy Evaluation (online / simulation)
-----------------------------------------
SuccessRateMetric     — binary task success/failure
TaskCompletionMetric  — partial sub-task completion
GoalReachingMetric    — end-effector distance to goal (with thresholded accuracy)
EpisodeLengthMetric   — efficiency: steps used vs episode budget

Example
-------
::

    from Halo_VLA.metrics import ActionL1Metric, SuccessRateMetric

    l1 = ActionL1Metric(action_dim=7)
    sr = SuccessRateMetric()

    # offline eval loop
    l1.reset()
    for pred, target in dataloader:
        l1.update(pred=pred, target=target)
    print(l1.compute())

    # policy eval loop
    sr.reset()
    for episode in sim_episodes:
        sr.update(success=run_episode(policy, env))
    print(sr.compute())
"""

from .base import RoboticsMetricBase

from .action import (
    ActionL1Metric,
    ActionMSEMetric,
    ActionChunkAccuracyMetric,
    TrajectorySmoothnessMetric,
    ActionNormalisedError,
)

from .generation import (
    PSNRMetric,
    SSIMMetric,
    FVDProxyMetric,
)

from .policy import (
    SuccessRateMetric,
    TaskCompletionMetric,
    GoalReachingMetric,
    EpisodeLengthMetric,
)

__all__ = [
    # base
    "RoboticsMetricBase",
    # action
    "ActionL1Metric",
    "ActionMSEMetric",
    "ActionChunkAccuracyMetric",
    "TrajectorySmoothnessMetric",
    "ActionNormalisedError",
    # generation
    "PSNRMetric",
    "SSIMMetric",
    "FVDProxyMetric",
    # policy
    "SuccessRateMetric",
    "TaskCompletionMetric",
    "GoalReachingMetric",
    "EpisodeLengthMetric",
]
