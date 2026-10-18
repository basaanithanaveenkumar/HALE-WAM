"""MomaConfig: a local AIRoA MoMa clone (``dataset: moma``)."""

from __future__ import annotations

from typing import Literal

from config.sections.base import VLASection


class MomaConfig(VLASection):
    """A local AIRoA MoMa clone (``dataset: moma``)."""

    ARGS = {
        "data_root": "moma_data_root",
        "camera": "moma_camera",
        "num_frames": "moma_num_frames",
        "frame_stride": "moma_frame_stride",
        "max_action_len": "moma_max_action_len",
        "num_workers": "moma_num_workers",
    }

    data_root: str | None = None
    camera: Literal["hand", "head"] = "head"
    num_frames: int = 5
    frame_stride: int = 25
    max_action_len: int = 256
    num_workers: int = 0  # video decoding is safest in the main process
