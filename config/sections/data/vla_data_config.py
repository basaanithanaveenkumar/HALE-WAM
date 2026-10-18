"""VLADataConfig: training data."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from config.sections.base import VLASection

from .droid_config import DroidConfig
from .moma_config import MomaConfig
from .web_dataset_section import WebDatasetSection


class VLADataConfig(VLASection):
    """Training data.

    ``dataset``: ``eo`` (EO-Data1.5M from the Hub, ``subset``), ``moma`` (``moma.data_root``)
    or ``droid`` (``droid.data_root``).
    """

    ARGS = {
        "dataset": "dataset",
        "subset": "subset",
        "num_workers": "num_workers",
        "max_seq_len": "max_seq_len",
        "max_samples": "max_samples",
    }

    dataset: Literal["eo", "moma", "droid"] = "eo"
    subset: str = "interleave-temporal"
    num_workers: int = 4
    max_seq_len: int = 256
    max_samples: int | None = 2000  # null = the whole dataset
    droid: DroidConfig = Field(default_factory=DroidConfig)
    moma: MomaConfig = Field(default_factory=MomaConfig)
    webdataset: WebDatasetSection = Field(default_factory=WebDatasetSection)
