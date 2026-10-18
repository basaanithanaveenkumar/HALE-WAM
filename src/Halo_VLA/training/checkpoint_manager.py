"""CheckpointManager: atomic checkpoint writes, pruning and restore for ``scripts/train.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from exceptions import CheckpointNotFoundError
from loguru import logger
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from config import HaloVLMConfig

#: Checkpoint file stem; the epoch number follows (``halo_vla_epoch12.pt``).
CHECKPOINT_PREFIX = "halo_vla_epoch"


class CheckpointManager:
    """Writes ``halo_vla_epoch<N>.pt`` files atomically and restores training state from them.

    A checkpoint is written to a temporary name and renamed (atomic on POSIX), so a full disk
    or a crash never leaves a corrupted file; afterwards only the newest ``keep`` are kept.
    Complexity: pruning sorts the directory listing, O(k log k) for k checkpoints.

    Args:
        directory: where checkpoints live (created on first use).
        keep: number of newest checkpoints to retain; ``0`` keeps all.

    Example:
        >>> manager = CheckpointManager(Path("checkpoints"), keep=3)  # doctest: +SKIP
        >>> manager.save(epoch, step, network, optimizer, scheduler, scaler, config)  # doctest: +SKIP
    """

    def __init__(self, directory: Path, keep: int = 0) -> None:
        self.directory = directory
        self.keep = keep

    def path_for(self, epoch: int) -> Path:
        """File of the checkpoint for ``epoch``."""
        return self.directory / f"{CHECKPOINT_PREFIX}{epoch}.pt"

    def existing(self) -> list[Path]:
        """Checkpoints on disk, oldest first."""
        return sorted(
            self.directory.glob(f"{CHECKPOINT_PREFIX}*.pt"),
            key=lambda p: int(p.stem.removeprefix(CHECKPOINT_PREFIX)),
        )

    def latest(self) -> Path | None:
        """The newest checkpoint, or ``None`` when there is none."""
        found = self.existing()
        return found[-1] if found else None

    def save(self, epoch: int, global_step: int, network: nn.Module, optimizer: Optimizer,
             scheduler: LRScheduler, scaler: GradScaler, config: HaloVLMConfig) -> Path:
        """Write the checkpoint for ``epoch`` and prune old ones.

        Args:
            epoch: finished epoch.
            global_step: optimiser steps taken.
            network: the unwrapped model (keys carry no DDP prefix).
            optimizer: optimiser whose state is saved.
            scheduler: learning-rate scheduler.
            scaler: AMP gradient scaler.
            config: the model configuration.

        Returns:
            The path written.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(epoch)
        tmp_path = self.directory / f".tmp_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
            },
            tmp_path,
        )
        tmp_path.replace(path)
        logger.info("Saved checkpoint -> {}", path)
        self.prune()
        return path

    def prune(self) -> None:
        """Delete all but the newest ``keep`` checkpoints (no-op when ``keep <= 0``)."""
        if self.keep <= 0:
            return
        for old in self.existing()[: -self.keep]:
            old.unlink()
            logger.info("Removed old checkpoint {}", old)

    @staticmethod
    def restore(path: Path, network: nn.Module, optimizer: Optimizer, scheduler: LRScheduler,
                scaler: GradScaler, device: torch.device) -> tuple[int, int]:
        """Load a checkpoint into the training objects.

        Args:
            path: checkpoint file.
            network: model to load the weights into.
            optimizer: optimiser to restore (kept fresh if its state does not match).
            scheduler: scheduler to restore.
            scaler: AMP scaler to restore when the checkpoint has one.
            device: where to map the tensors.

        Returns:
            ``(start_epoch, global_step)`` to continue from.

        Raises:
            CheckpointNotFoundError: if ``path`` does not exist.
        """
        if not path.exists():
            raise CheckpointNotFoundError(f"Checkpoint not found: {path}")
        logger.info("Resuming from checkpoint: {}", path)
        with torch.serialization.safe_globals([HaloVLMConfig]):
            ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        network.load_state_dict(ckpt["model_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            # Checkpoints from before the trainable-only optimizer hold state for every
            # parameter; keep the weights and restart the optimizer moments.
            logger.warning("optimizer state not restored ({}); continuing with fresh moments", exc)
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed at epoch {} / global_step {}  (training from epoch {})",
                    ckpt.get("epoch"), global_step, start_epoch)
        return start_epoch, global_step
