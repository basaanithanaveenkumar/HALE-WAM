"""Smoke demo: ``python -m eo_dataset`` (formerly the module's ``__main__`` block)."""

from __future__ import annotations

import argparse

import torch
from loguru import logger

from . import *  # noqa: F403

parser = argparse.ArgumentParser(description="EO-Data1.5M dataloader test")
parser.add_argument("--subset", default="interleave-temporal")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_seq_len", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()

loader = build_eo_dataloader(
    subset=args.subset,
    batch_size=args.batch_size,
    max_seq_len=args.max_seq_len,
    num_workers=args.num_workers,
    shuffle=False,
)

logger.info(f"\nDataset size: {len(loader.dataset)}")
logger.info(f"Batches:      {len(loader)}\n")

batch = next(iter(loader))

for key, val in batch.items():
    if isinstance(val, torch.Tensor):
        logger.info(f"  {key:20s} → {val.shape}  dtype={val.dtype}")
    else:
        logger.info(f"  {key:20s} → {type(val).__name__}: {val}")

logger.info("\n✓ Dataloader works correctly.")
