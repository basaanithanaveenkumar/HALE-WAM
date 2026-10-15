"""``logging`` and ``viz``: console / TensorBoard logging and training GIFs."""

from __future__ import annotations

from config.sections.base import VLASection


class VLALoggingConfig(VLASection):
    ARGS = {"log_every": "log_every", "tensorboard_dir": "tensorboard_dir", "tb_run_name": "tb_run_name",
            "tb_image_every": "tb_image_every"}

    log_every: int = 10
    tensorboard_dir: str = "runs"          # "" disables TensorBoard
    tb_run_name: str = ""                  # default: timestamp
    tb_image_every: int = 0


class VLAVizConfig(VLASection):
    ARGS = {"every_n_steps": "vis_every", "max_videos": "vis_max_videos", "min_frames": "vis_min_frames",
            "fps_future_frames": "vis_fps_future_frames"}

    every_n_steps: int = 10                # 0 = disabled
    max_videos: int = 4
    min_frames: int = 3
    fps_future_frames: int = 2
