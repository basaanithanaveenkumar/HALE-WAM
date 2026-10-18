"""VLAVizConfig: training-time visualisation GIFs."""

from __future__ import annotations

from config.sections.base import VLASection


class VLAVizConfig(VLASection):
    """Visualisation GIFs written during training.

    Attributes:
        every_n_steps: optimiser steps between GIFs (0 disables).
        max_videos: samples visualised per event.
        min_frames: minimum frames a sample needs to be shown.
        fps_future_frames: frame rate of the predicted-future strip.
    """

    ARGS = {"every_n_steps": "vis_every", "max_videos": "vis_max_videos", "min_frames": "vis_min_frames",
            "fps_future_frames": "vis_fps_future_frames"}

    every_n_steps: int = 10                # 0 = disabled
    max_videos: int = 4
    min_frames: int = 3
    fps_future_frames: int = 2
