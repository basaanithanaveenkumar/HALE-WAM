"""Hold out a reproducible validation subset from the training dataset."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, IterableDataset, Subset


def split_validation(loader: DataLoader, fraction: float, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    """Split ``loader``'s map-style dataset into train / validation loaders.

    The split is a fixed random permutation (``seed``), so every rank and every resumed run
    holds out the same samples. The validation loader is unshuffled and keeps the batch size,
    workers and collate function of ``loader``.

    Args:
        loader: the training loader built by a dataset builder.
        fraction: share of samples held out, in ``(0, 1)``.
        seed: permutation seed.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {fraction}")
    dataset = loader.dataset
    if isinstance(dataset, IterableDataset):
        raise ValueError("a validation split needs a map-style dataset (not a stream)")
    n = len(dataset)
    n_val = max(1, int(round(n * fraction)))
    if n_val >= n:
        raise ValueError(f"val_fraction={fraction} leaves no training samples out of {n}")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_set, val_set = Subset(dataset, order[n_val:]), Subset(dataset, order[:n_val])
    common = dict(batch_size=loader.batch_size, num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                  pin_memory=loader.pin_memory)
    return (DataLoader(train_set, shuffle=True, drop_last=loader.drop_last, **common),
            DataLoader(val_set, shuffle=False, **common))
