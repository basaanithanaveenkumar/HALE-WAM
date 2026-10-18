"""CheckpointManager: atomic writes, pruning, latest lookup and restore."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import HaloVLMConfig
from exceptions import CheckpointNotFoundError
from training import CheckpointManager


def make_training_objects():
    network = nn.Linear(3, 2)
    optimizer = AdamW(network.parameters(), lr=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    return network, optimizer, scheduler, GradScaler("cpu", enabled=False)


def save_epochs(manager: CheckpointManager, epochs, objects) -> None:
    network, optimizer, scheduler, scaler = objects
    for epoch in epochs:
        manager.save(epoch, epoch * 10, network, optimizer, scheduler, scaler, HaloVLMConfig())


def test_save_prunes_to_the_newest_keep_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "ckpt", keep=2)
    save_epochs(manager, [1, 2, 3, 10], make_training_objects())
    assert [p.name for p in manager.existing()] == ["halo_vla_epoch3.pt", "halo_vla_epoch10.pt"]
    assert manager.latest() == manager.path_for(10)
    assert not list((tmp_path / "ckpt").glob(".tmp*"))


def test_keep_zero_keeps_everything_and_latest_of_an_empty_directory_is_none(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path, keep=0)
    assert manager.latest() is None
    save_epochs(manager, [1, 2, 3], make_training_objects())
    assert len(manager.existing()) == 3


def test_restore_returns_the_epoch_to_continue_from_and_loads_the_weights(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    saved = make_training_objects()
    save_epochs(manager, [4], saved)
    network, optimizer, scheduler, scaler = make_training_objects()
    start_epoch, step = CheckpointManager.restore(
        manager.path_for(4), network, optimizer, scheduler, scaler, torch.device("cpu")
    )
    assert (start_epoch, step) == (5, 40)
    assert torch.equal(network.weight, saved[0].weight)


def test_restore_of_a_missing_file_raises_a_checkpoint_error(tmp_path: Path) -> None:
    network, optimizer, scheduler, scaler = make_training_objects()
    with pytest.raises(CheckpointNotFoundError, match="not found"):
        CheckpointManager.restore(
            tmp_path / "nope.pt", network, optimizer, scheduler, scaler, torch.device("cpu")
        )
