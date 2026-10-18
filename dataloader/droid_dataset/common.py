"""Helpers shared by the ``droid_dataset`` package."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from dataloader.eo_dataset import eo_collate_fn

from .droid_config import DROID_ACTION_DIM, DROID_STATE_DIM, DROIDConfig
from .droid_dataset import DROIDDataset

# ---------------------------------------------------------------------------
# Known DROID camera view keys (lerobot v2 naming)
# ---------------------------------------------------------------------------
DROID_CAMERA_KEYS = [
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
]


def build_droid_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 0,
    tokenizer: Any = None,
    img_size: int = 224,
    max_seq_len: int = 512,
    max_action_len: int = 256,
    action_dim: int = DROID_ACTION_DIM,
    state_dim: int = DROID_STATE_DIM,
    camera: str = "observation.images.exterior_image_1_left",
    num_sample_frames: int = 5,
    num_predict_frames: int = 5,
    frame_stride: int = 6,
    split: str = "train",
    shuffle: bool = True,
    max_samples: int | None = None,
    pin_memory: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """
    Build a DataLoader for DROID compatible with ``scripts/train.py``.

    Keys returned per batch match those of the MoMa and EO loaders so the
    training loop needs no changes.

    Args:
        data_root: Local path to the lerobot/droid_100 clone.
        num_workers: Set to 0 if PyAV is not fork-safe on your system.
        camera: Which camera view to use for image frames.
            Options: ``observation.images.exterior_image_1_left`` (default),
            ``observation.images.exterior_image_2_left``,
            ``observation.images.wrist_image_left``.
        num_sample_frames: Context frames given to the model.
        num_predict_frames: Ground-truth future frames returned for the DiT head.
        frame_stride: Temporal gap between sampled frames (larger = more motion).
    """
    cfg = DROIDConfig(
        data_root=data_root,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        state_dim=state_dim,
        camera=camera,
        num_sample_frames=num_sample_frames,
        num_predict_frames=num_predict_frames,
        frame_stride=frame_stride,
        max_samples=max_samples,
        **{k: v for k, v in kwargs.items() if k in DROIDConfig.__dataclass_fields__},
    )
    ds = DROIDDataset(config=cfg, tokenizer=tokenizer, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and split == "train",
        num_workers=num_workers,
        collate_fn=eo_collate_fn,
        pin_memory=pin_memory and num_workers > 0,
        drop_last=len(ds) > batch_size,
    )
