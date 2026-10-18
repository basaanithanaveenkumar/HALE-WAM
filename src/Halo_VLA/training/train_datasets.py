"""Training-dataset registry: ``--dataset eo | droid | moma`` -> a train DataLoader.

Pattern: Registry. A builder takes the parsed arguments, the model config and the (optional) LLM
tokenizer and returns a ``DataLoader``. Add a dataset by registering a builder; the CLI choices
and ``scripts/train.py`` need no change.

Example:
    >>> loader = TRAIN_DATASETS.get(args.dataset)(args, config, tokenizer)  # doctest: +SKIP
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from hale_core.exceptions import RegistryError
from hale_core.registry import NamedRegistry
from torch.utils.data import DataLoader

from config import HaloVLMConfig
from exceptions import HaloVLAConfigError

#: ``dataset`` name -> ``(args, config, tokenizer) -> DataLoader``.
TRAIN_DATASETS = NamedRegistry("training dataset")

TrainDatasetBuilder = Callable[[argparse.Namespace, HaloVLMConfig, Any], DataLoader]


@TRAIN_DATASETS.register_module(name="eo")
def build_eo_train_loader(
    args: argparse.Namespace, config: HaloVLMConfig, tokenizer: Any
) -> DataLoader:
    """IPEC-COMMUNITY/EO-Data1.5M (``--subset``)."""
    from dataloader.eo_dataset import build_eo_dataloader  # lazy: datasets / video decoders

    return build_eo_dataloader(
        subset=args.subset,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=config.img_size,
        max_seq_len=args.max_seq_len,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        num_predict_frames=args.num_predict_frames,
        shuffle=True,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
    )


@TRAIN_DATASETS.register_module(name="droid")
def build_droid_train_loader(
    args: argparse.Namespace, config: HaloVLMConfig, tokenizer: Any
) -> DataLoader:
    """DROID episodes under ``--droid_data_root``."""
    from dataloader.droid_dataset import build_droid_dataloader  # lazy

    if not args.droid_data_root:
        raise HaloVLAConfigError("--droid_data_root is required when --dataset droid")
    return build_droid_dataloader(
        data_root=args.droid_data_root,
        batch_size=args.batch_size,
        num_workers=args.moma_num_workers,
        img_size=config.img_size,
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


@TRAIN_DATASETS.register_module(name="moma")
def build_moma_train_loader(
    args: argparse.Namespace, config: HaloVLMConfig, tokenizer: Any
) -> DataLoader:
    """AIROA MoMa episodes under ``--moma_data_root``."""
    from dataloader.airoa_moma_dataset import build_airoa_moma_dataloader  # lazy

    if not args.moma_data_root:
        raise HaloVLAConfigError("--moma_data_root is required when --dataset moma")
    return build_airoa_moma_dataloader(
        data_root=args.moma_data_root,
        batch_size=args.batch_size,
        num_workers=args.moma_num_workers,
        img_size=config.img_size,
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


def build_train_loader(
    args: argparse.Namespace, config: HaloVLMConfig, tokenizer: Any
) -> DataLoader:
    """The train loader of ``args.dataset``.

    Args:
        args: parsed training arguments.
        config: model configuration (image size).
        tokenizer: the LLM backbone's tokenizer, or ``None`` for the datasets' default.

    Returns:
        The DataLoader.

    Raises:
        HaloVLAConfigError: for an unknown dataset or a missing data root.
    """
    try:
        builder = TRAIN_DATASETS.get(args.dataset)
    except RegistryError as err:
        raise HaloVLAConfigError(
            f"unknown --dataset {args.dataset!r}; choose one of {sorted(TRAIN_DATASETS.keys())}"
        ) from err
    return builder(args, config, tokenizer)
