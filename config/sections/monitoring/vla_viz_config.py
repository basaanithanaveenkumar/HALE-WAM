"""VLAVizConfig: vLAVizConfig."""

from __future__ import annotations

from config.sections.base import VLASection


class VLAVizConfig(VLASection):
    ARGS = {"every_n_steps": "vis_every", "max_videos": "vis_max_videos", "min_frames": "vis_min_frames",
            "fps_future_frames": "vis_fps_future_frames"}

    every_n_steps: int = 10                # 0 = disabled
    max_videos: int = 4
    min_frames: int = 3
    fps_future_frames: int = 2
