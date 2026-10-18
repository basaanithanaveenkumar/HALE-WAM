"""VLALoggingConfig: console and TensorBoard logging cadence."""

from __future__ import annotations

from config.sections.base import VLASection


class VLALoggingConfig(VLASection):
    """Console and TensorBoard logging.

    Attributes:
        log_every: optimiser steps between log lines.
        tensorboard_dir: TensorBoard root (empty disables TensorBoard).
        tb_run_name: sub-directory name (default: a timestamp).
        tb_image_every: steps between input-frame images (0 disables).
    """

    ARGS = {"log_every": "log_every", "tensorboard_dir": "tensorboard_dir", "tb_run_name": "tb_run_name",
            "tb_image_every": "tb_image_every"}

    log_every: int = 10
    tensorboard_dir: str = "runs"          # "" disables TensorBoard
    tb_run_name: str = ""                  # default: timestamp
    tb_image_every: int = 0
