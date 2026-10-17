"""VLALoggingConfig: vLALoggingConfig."""

from __future__ import annotations

from config.sections.base import VLASection


class VLALoggingConfig(VLASection):
    ARGS = {"log_every": "log_every", "tensorboard_dir": "tensorboard_dir", "tb_run_name": "tb_run_name",
            "tb_image_every": "tb_image_every"}

    log_every: int = 10
    tensorboard_dir: str = "runs"          # "" disables TensorBoard
    tb_run_name: str = ""                  # default: timestamp
    tb_image_every: int = 0
