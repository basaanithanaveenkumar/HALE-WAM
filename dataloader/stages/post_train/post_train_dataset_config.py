"""PostTrainDatasetConfig: postTrainDatasetConfig."""

from __future__ import annotations

from dataclasses import dataclass, field

from .post_train_dataset_registry import POST_TRAIN_DATASETS

POST_TRAIN_DATASET_NAMES: tuple[str, ...] = tuple(POST_TRAIN_DATASETS)


@dataclass
class PostTrainDatasetConfig:
    """Configuration for the post-training dataloader."""

    dataset_names: list[str] = field(default_factory=lambda: list(POST_TRAIN_DATASET_NAMES))
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32
    # For task-filtered datasets supply a keyword (e.g. "pick_and_place").
    task_filter: str | None = None
    max_samples_per_dataset: int | None = None  # small datasets — keep all
    streaming: bool = False  # materialise in memory for small task sets
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    # Local paths for gated / large datasets
    droid_data_root: str | None = None
    airoa_data_root: str | None = None
    # Local-collection datasets (preference / DAgger / hindsight)
    flowpro_data_root: str | None = None
    apo_data_root: str | None = None
    hindsight_data_root: str | None = None
    dagger_data_root: str | None = None
