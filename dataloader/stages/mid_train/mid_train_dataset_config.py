"""MidTrainDatasetConfig: midTrainDatasetConfig."""

from __future__ import annotations

from dataclasses import dataclass, field

from .mid_train_dataset_registry import MID_TRAIN_DATASETS

MID_TRAIN_DATASET_NAMES: tuple[str, ...] = tuple(MID_TRAIN_DATASETS)


@dataclass
class MidTrainDatasetConfig:
    """Configuration for the mid-training dataloader."""

    dataset_names: list[str] = field(default_factory=lambda: list(MID_TRAIN_DATASET_NAMES))
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32
    max_samples_per_dataset: int | None = 50_000
    streaming: bool = True
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    # DROID-specific
    droid_data_root: str | None = None
    droid_num_sample_frames: int = 1
    droid_num_predict_frames: int = 1
    # AIRoA-MoMA-specific
    airoa_data_root: str | None = None
