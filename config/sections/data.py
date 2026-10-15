"""``data``: which robot dataset feeds training and how (EO-Data, AIRoA MoMa, DROID, WebDataset)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from config.sections.base import VLASection


class DroidConfig(VLASection):
    """A local ``lerobot/droid_100`` clone (``dataset: droid``)."""

    ARGS = {"data_root": "droid_data_root", "camera": "droid_camera"}

    data_root: str | None = None
    camera: Literal["observation.images.exterior_image_1_left", "observation.images.exterior_image_2_left",
                    "observation.images.wrist_image_left"] = "observation.images.exterior_image_1_left"


class MomaConfig(VLASection):
    """A local AIRoA MoMa clone (``dataset: moma``)."""

    ARGS = {"data_root": "moma_data_root", "camera": "moma_camera", "num_frames": "moma_num_frames",
            "frame_stride": "moma_frame_stride", "max_action_len": "moma_max_action_len",
            "num_workers": "moma_num_workers"}

    data_root: str | None = None
    camera: Literal["hand", "head"] = "head"
    num_frames: int = 5
    frame_stride: int = 25
    max_action_len: int = 256
    num_workers: int = 0                   # video decoding is safest in the main process


class WebDatasetSection(VLASection):
    """Convert the dataset to WebDataset tar shards once, then stream them (docs/distributed.md)."""

    ARGS = {"enabled": "webdataset", "root": "webdataset_root", "samples_per_shard": "webdataset_samples_per_shard",
            "convert_workers": "webdataset_convert_workers", "shuffle_buffer": "webdataset_shuffle_buffer",
            "overwrite": "webdataset_overwrite"}

    enabled: bool = False
    root: str = "data/webdataset"
    samples_per_shard: int = 256
    convert_workers: int = 0
    shuffle_buffer: int = 1000
    overwrite: bool = False


class VLADataConfig(VLASection):
    """Training data.

    ``dataset``: ``eo`` (EO-Data1.5M from the Hub, ``subset``), ``moma`` (``moma.data_root``)
    or ``droid`` (``droid.data_root``).
    """

    ARGS = {"dataset": "dataset", "subset": "subset", "num_workers": "num_workers", "max_seq_len": "max_seq_len",
            "max_samples": "max_samples"}

    dataset: Literal["eo", "moma", "droid"] = "eo"
    subset: str = "interleave-temporal"
    num_workers: int = 4
    max_seq_len: int = 256
    max_samples: int | None = 2000         # null = the whole dataset
    droid: DroidConfig = Field(default_factory=DroidConfig)
    moma: MomaConfig = Field(default_factory=MomaConfig)
    webdataset: WebDatasetSection = Field(default_factory=WebDatasetSection)
