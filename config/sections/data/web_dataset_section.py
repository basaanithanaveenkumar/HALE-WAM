"""WebDatasetSection: convert the dataset to WebDataset tar shards once, then stream them (docs/distributed.md)."""

from __future__ import annotations

from config.sections.base import VLASection


class WebDatasetSection(VLASection):
    """Convert the dataset to WebDataset tar shards once, then stream them (docs/distributed.md)."""

    ARGS = {
        "enabled": "webdataset",
        "root": "webdataset_root",
        "samples_per_shard": "webdataset_samples_per_shard",
        "convert_workers": "webdataset_convert_workers",
        "shuffle_buffer": "webdataset_shuffle_buffer",
        "overwrite": "webdataset_overwrite",
    }

    enabled: bool = False
    root: str = "data/webdataset"
    samples_per_shard: int = 256
    convert_workers: int = 0
    shuffle_buffer: int = 1000
    overwrite: bool = False
