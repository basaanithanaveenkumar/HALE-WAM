"""Helpers shared by the ``airoa_moma_dataset`` package."""

from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader

from dataloader.eo_dataset import eo_collate_fn

from .airoa_moma_config import AiroaMomaConfig
from .airoa_moma_dataset import AiroaMomaDataset


def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        from hale_core.exceptions import MissingDependencyError

        raise MissingDependencyError("cv2", "OpenCV decodes MoMa video: pip install opencv-python") from e


def build_airoa_moma_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    tokenizer=None,
    img_size: int = 224,
    max_seq_len: int = 512,
    max_action_len: int = 256,
    action_dim: int = 32,
    state_dim: int = 32,
    num_sample_frames: int = 8,
    num_predict_frames: int = 1,
    frame_stride: int = 4,
    camera: str = "head",
    split: str = "train",
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a DataLoader compatible with ``scripts/train.py`` (same keys as EO loader).

    Args:
        data_root: Local clone root (``episodes.jsonl`` + ``videos/``).
        num_predict_frames: Ground-truth future frames returned as ``future_frames``
            in each batch item (separate from the ``num_sample_frames`` context frames).
    """
    cfg = AiroaMomaConfig(
        data_root=data_root,
        img_size=img_size,
        max_seq_len=max_seq_len,
        max_action_len=max_action_len,
        action_dim=action_dim,
        state_dim=state_dim,
        num_sample_frames=num_sample_frames,
        num_predict_frames=num_predict_frames,
        frame_stride=frame_stride,
        camera=camera,
        max_samples=max_samples,
        **{k: v for k, v in kwargs.items() if k in AiroaMomaConfig.__dataclass_fields__},
    )
    ds = AiroaMomaDataset(config=cfg, tokenizer=tokenizer, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=eo_collate_fn,
        pin_memory=pin_memory,
        drop_last=len(ds) > batch_size,   # never drop if there is only 1 batch
    )
