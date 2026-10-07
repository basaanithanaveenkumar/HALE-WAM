"""Stage-aware dataloaders for Halo-VLA: pretrain / mid-train / post-train.

Usage::

    from dataloader.stages import TrainingStage, build_stage_dataloader
    from dataloader.stages.pretrain import PretrainDatasetRegistry
    from dataloader.stages.mid_train import MidTrainDatasetRegistry
    from dataloader.stages.post_train import PostTrainDatasetRegistry

    loader = build_stage_dataloader(TrainingStage.PRETRAIN, batch_size=32)
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TrainingStage(StrEnum):
    """High-level training phase for Halo-VLA policies.

      PRETRAIN   — web-scale video/image data + large open robot teleoperation
      MID_TRAIN  — domain-specific robot manipulation (lab-collected corpora)
      POST_TRAIN — task-specific fine-tuning on target robot and task
    """

    PRETRAIN = "pretrain"
    MID_TRAIN = "mid_train"
    POST_TRAIN = "post_train"


def build_stage_dataloader(
    stage: TrainingStage,
    *,
    batch_size: int = 8,
    num_workers: int = 2,
    img_size: int = 224,
    max_seq_len: int = 77,
    max_action_len: int = 64,
    action_dim: int = 32,
    **dataset_kwargs,
) -> "DataLoader":
    """Return a DataLoader for the given training stage.

    Args:
        stage: One of TrainingStage.PRETRAIN / MID_TRAIN / POST_TRAIN.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        img_size: Resize target for all image frames.
        max_seq_len: Token length cap for text sequences.
        max_action_len: Maximum action sequence length.
        action_dim: Dimensionality of the action vector.
        **dataset_kwargs: Forwarded verbatim to the stage dataset class.

    Returns:
        A configured ``torch.utils.data.DataLoader``.
    """
    from dataloader.stages.pretrain import PretrainDatasetRegistry
    from dataloader.stages.mid_train import MidTrainDatasetRegistry
    from dataloader.stages.post_train import PostTrainDatasetRegistry

    _registries = {
        TrainingStage.PRETRAIN: PretrainDatasetRegistry,
        TrainingStage.MID_TRAIN: MidTrainDatasetRegistry,
        TrainingStage.POST_TRAIN: PostTrainDatasetRegistry,
    }
    registry_cls = _registries[stage]
    return registry_cls(
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        **dataset_kwargs,
    ).build()


__all__ = [
    "TrainingStage",
    "build_stage_dataloader",
]
