"""DroidConfig: a local ``lerobot/droid_100`` clone (``dataset: droid``)."""

from __future__ import annotations

from typing import Literal

from config.sections.base import VLASection


class DroidConfig(VLASection):
    """A local ``lerobot/droid_100`` clone (``dataset: droid``)."""

    ARGS = {"data_root": "droid_data_root", "camera": "droid_camera"}

    data_root: str | None = None
    camera: Literal[
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
        "observation.images.wrist_image_left",
    ] = "observation.images.exterior_image_1_left"
