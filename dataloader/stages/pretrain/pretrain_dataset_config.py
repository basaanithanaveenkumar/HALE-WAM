"""PretrainDatasetConfig: pretrainDatasetConfig."""

from __future__ import annotations

from dataclasses import dataclass, field

from .pretrain_dataset_registry import PRETRAIN_DATASETS

PRETRAIN_DATASET_NAMES: tuple[str, ...] = tuple(PRETRAIN_DATASETS)


@dataclass
class PretrainDatasetConfig:
    """Configuration for the pre-training dataloader."""

    dataset_names: list[str] = field(default_factory=lambda: list(PRETRAIN_DATASET_NAMES))
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32
    max_samples_per_dataset: int | None = None  # None = no cap
    streaming: bool = True
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
