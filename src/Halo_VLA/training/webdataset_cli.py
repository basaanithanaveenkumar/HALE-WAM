"""``--webdataset*`` command-line flags of ``scripts/train.py``: any dataset -> WebDataset shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from torch.utils.data import DataLoader

#: CLI arguments that change the samples a dataset produces (they form the shard fingerprint).
DATA_ARGUMENTS = (
    "dataset", "subset", "moma_data_root", "droid_data_root", "moma_camera", "droid_camera",
    "moma_num_frames", "moma_frame_stride", "moma_max_action_len", "max_seq_len", "action_dim",
    "state_dim", "action_chunk_size", "num_predict_frames", "max_samples",
    # the LLM backbone's tokenizer defines the token ids stored in the shards
    "llm_backbone", "llm_model_id", "llm_checkpoint",
    # the validation split removes samples from the training shards
    "val_fraction", "val_seed",
)


class WebDatasetCliOptions:
    """Adds the WebDataset flags and swaps a DataLoader's dataset for its shard stream.

    With ``--webdataset`` the first run writes the dataset (whatever ``--dataset`` built: EO,
    DROID, AIRoA MoMa) to tar shards under ``--webdataset_root``; later runs - and every
    distributed rank - stream the shards instead of decoding videos / parquet again.
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Add the ``--webdataset*`` flags to ``parser``."""
        group = parser.add_argument_group("WebDataset (optional, needs hale-blocks[webdataset])")
        group.add_argument("--webdataset", action="store_true",
                           help="Convert the training dataset to WebDataset shards and stream them")
        group.add_argument("--webdataset_root", default="data/webdataset", help="Shard root directory")
        group.add_argument("--webdataset_samples_per_shard", type=int, default=256,
                           help="Samples per tar shard (keep >= ranks x DataLoader workers shards)")
        group.add_argument("--webdataset_convert_workers", type=int, default=0,
                           help="DataLoader workers reading the source dataset while converting")
        group.add_argument("--webdataset_shuffle_buffer", type=int, default=1000,
                           help="In-memory sample shuffle buffer while training")
        group.add_argument("--webdataset_overwrite", action="store_true", help="Re-convert existing shards")

    @staticmethod
    def build_config(args: argparse.Namespace) -> Any:
        """The ``hale_core`` ``WebDatasetConfig`` for ``args`` (imports HaleBlocks lazily)."""
        try:
            from hale_core.config.sections.webdataset import WebDatasetConfig
        except ImportError as exc:  # pragma: no cover - depends on the environment
            from hale_core.exceptions import MissingDependencyError

            raise MissingDependencyError(
                "hale_core.config.sections.webdataset",
                "--webdataset needs HaleBlocks: pip install -e '.[webdataset]'",
            ) from exc
        return WebDatasetConfig(
            enabled=args.webdataset,
            root=args.webdataset_root,
            samples_per_shard=args.webdataset_samples_per_shard,
            convert_workers=args.webdataset_convert_workers,
            shuffle_buffer=args.webdataset_shuffle_buffer,
            overwrite=args.webdataset_overwrite,
        )

    @staticmethod
    def fingerprint(args: argparse.Namespace) -> str:
        """Hash of the arguments that change the dataset's samples."""
        spec = {name: getattr(args, name, None) for name in DATA_ARGUMENTS}
        # Image size can come from the YAML / --model_option instead of a flag (only added when
        # set, so shards written before these options existed keep their fingerprint).
        from config.overrides import parse_model_options

        overrides = {**(getattr(args, "model_overrides", None) or {}),
                     **parse_model_options(getattr(args, "model_option", None))}
        if overrides.get("img_size") is not None:
            spec["img_size"] = overrides["img_size"]
        return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()

    @classmethod
    def prepare_loader(cls, loader: DataLoader, args: argparse.Namespace) -> DataLoader:
        """``loader`` unchanged without ``--webdataset``; otherwise the same loader (batch size,
        collate function, workers) over the dataset's WebDataset stream."""
        if not args.webdataset:
            return loader
        from hale_core.data.webdataset import WebDatasetAdapter

        fingerprint = cls.fingerprint(args)
        stream = WebDatasetAdapter(cls.build_config(args)).prepare(
            loader.dataset,
            f"{args.dataset}-train-{fingerprint[:12]}",
            training=True,
            loader_workers=loader.num_workers,
            fingerprint=fingerprint,
        )
        return DataLoader(
            stream,
            batch_size=loader.batch_size,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )
