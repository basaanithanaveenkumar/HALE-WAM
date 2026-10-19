"""Training dataloaders selectable by ``--dataset`` (EO-Data1.5M, DROID, AIRoA MoMa)."""

from __future__ import annotations

import argparse
from typing import Any

from exceptions import HaloVLAConfigError
from hale_core.registry import NamedRegistry
from torch.utils.data import DataLoader

from dataloader.airoa_moma_dataset import build_airoa_moma_dataloader
from dataloader.droid_dataset import build_droid_dataloader
from dataloader.eo_dataset import build_eo_dataloader

#: ``--dataset`` -> builder ``(args, img_size, tokenizer) -> DataLoader``.
TRAIN_DATALOADERS = NamedRegistry("training dataset")


@TRAIN_DATALOADERS.register_module(name="eo")
def _build_eo_loader(args: argparse.Namespace, img_size: int, tokenizer: Any) -> DataLoader:
    """EO-Data1.5M from the Hugging Face Hub (``--subset``)."""
    return build_eo_dataloader(
        subset=args.subset,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        max_seq_len=args.max_seq_len,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        num_predict_frames=args.num_predict_frames,
        shuffle=True,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
    )


@TRAIN_DATALOADERS.register_module(name="droid")
def _build_droid_loader(args: argparse.Namespace, img_size: int, tokenizer: Any) -> DataLoader:
    """DROID in LeRobot format from ``--droid_data_root``."""
    if not args.droid_data_root:
        raise HaloVLAConfigError("--droid_data_root is required when --dataset droid")
    return build_droid_dataloader(
        data_root=args.droid_data_root,
        batch_size=args.batch_size,
        num_workers=args.moma_num_workers,
        img_size=img_size,
        max_seq_len=args.max_seq_len,
        max_action_len=args.moma_max_action_len,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        camera=args.droid_camera,
        num_sample_frames=args.moma_num_frames,
        num_predict_frames=args.num_predict_frames,
        frame_stride=args.moma_frame_stride,
        shuffle=True,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
    )


@TRAIN_DATALOADERS.register_module(name="moma")
def _build_moma_loader(args: argparse.Namespace, img_size: int, tokenizer: Any) -> DataLoader:
    """AIRoA MoMa from a local clone at ``--moma_data_root``."""
    if not args.moma_data_root:
        raise HaloVLAConfigError("--moma_data_root is required when --dataset moma")
    moma_workers = args.moma_num_workers
    return build_airoa_moma_dataloader(
        data_root=args.moma_data_root,
        batch_size=args.batch_size,
        num_workers=moma_workers,
        img_size=img_size,
        max_seq_len=args.max_seq_len,
        max_action_len=args.moma_max_action_len,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        num_sample_frames=args.moma_num_frames,
        num_predict_frames=args.num_predict_frames,
        frame_stride=args.moma_frame_stride,
        camera=args.moma_camera,
        shuffle=True,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
    )


def build_train_dataloader(args: argparse.Namespace, img_size: int, tokenizer: Any = None) -> DataLoader:
    """The training loader for ``args.dataset``.

    Args:
        args: parsed ``scripts/train.py`` arguments.
        img_size: model input resolution.
        tokenizer: the pretrained LLM's tokenizer, or ``None`` for the datasets' default.

    Raises:
        HaloVLAConfigError: for an unregistered dataset or a missing data root.
    """
    if args.dataset not in TRAIN_DATALOADERS:
        raise HaloVLAConfigError(f"unknown --dataset {args.dataset!r}; choose one of {list(TRAIN_DATALOADERS)}")
    return TRAIN_DATALOADERS.get(args.dataset)(args, img_size, tokenizer)
