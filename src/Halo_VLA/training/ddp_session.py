"""Process group, device and data sharding for single-process, torchrun and Ray runs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DDPSession:
    """Everything the training loop needs to know about its place in a distributed run.

    * **Single process** (no ``WORLD_SIZE`` > 1): rank 0 of 1, ``wrap`` returns the module as is.
    * **torchrun**: ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` are set; :meth:`start` creates the
      process group (NCCL on CUDA, Gloo otherwise).
    * **Ray Train**: Ray has already created the process group; :meth:`start` just uses it.

    Attributes:
        rank: global rank.
        local_rank: rank on this node (selects the CUDA device).
        world_size: number of processes.
        device: device this rank trains on.
        owns_group: whether :meth:`start` created the process group (and :meth:`close` destroys it).
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    owns_group: bool = False

    # ------------------------------------------------------------------ construction
    @classmethod
    def start(cls, device: str) -> DDPSession:
        """Join (or create) the process group and pick this rank's device.

        Args:
            device: the requested device (``--device``); with CUDA and several ranks each rank
                uses ``cuda:<local_rank>``.
        """
        owns = False
        world = int(os.environ.get("WORLD_SIZE", 1))
        if world > 1 and dist.is_available() and not dist.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            owns = True
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        else:
            rank, world = 0, 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        resolved = torch.device(device)
        if resolved.type == "cuda" and world > 1:
            resolved = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(resolved)
        return cls(rank=rank, local_rank=local_rank, world_size=world, device=resolved, owns_group=owns)

    # ------------------------------------------------------------------ properties
    @property
    def is_main(self) -> bool:
        """Rank 0 logs, writes TensorBoard / GIFs and saves checkpoints."""
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    # ------------------------------------------------------------------ model / data
    def wrap(self, module: nn.Module, find_unused_parameters: bool = False) -> nn.Module:
        """``DistributedDataParallel(module)`` when distributed, else ``module`` unchanged."""
        if not self.distributed:
            return module
        ids = [self.device.index] if self.device.type == "cuda" else None
        return DistributedDataParallel(
            module, device_ids=ids, find_unused_parameters=find_unused_parameters
        )

    def shard_loader(self, loader: DataLoader, shuffle: bool = True) -> DataLoader:
        """Give this rank a disjoint 1/world_size of a map-style dataset.

        Rebuilds the DataLoader with a :class:`DistributedSampler` (same batch size, collate,
        workers). Streams (``IterableDataset``, e.g. WebDataset) split themselves and are
        returned unchanged, as is everything in a single-process run.
        """
        if not self.distributed or isinstance(loader.dataset, IterableDataset):
            return loader
        sampler = DistributedSampler(
            loader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
        )
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            persistent_workers=loader.persistent_workers,
        )

    @staticmethod
    def set_epoch(loader: DataLoader, epoch: int) -> None:
        """Reshuffle a DistributedSampler for ``epoch`` (no-op for other samplers)."""
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # ------------------------------------------------------------------ collectives
    def mean(self, value: float) -> float:
        """Average a scalar over all ranks (for epoch summaries)."""
        if not self.distributed:
            return value
        t = torch.tensor([value], device=self.device, dtype=torch.float64)
        dist.all_reduce(t)
        return t.item() / self.world_size

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        """Destroy the process group if this session created it (torchrun)."""
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
