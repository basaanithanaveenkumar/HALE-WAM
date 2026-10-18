"""``scripts/train.py`` end to end: single process, WebDataset, DDP (2 ranks, gloo) and Ray Train.

The real ``train(args)`` runs with a tiny stand-in model and dataset (``tests/training_support``),
so every code path of the loop - session, DDP wrapping, loader sharding, shard conversion,
checkpointing and Ray reporting - is exercised on a CPU in seconds.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.training_support import TinyHaloVLM, load_train_script, tiny_args

HAS_HALE_CORE = importlib.util.find_spec("hale_core") is not None
HAS_WEBDATASET = HAS_HALE_CORE and importlib.util.find_spec("webdataset") is not None
HAS_RAY = HAS_HALE_CORE and importlib.util.find_spec("ray") is not None


def _checkpoints(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("halo_vla_epoch*.pt"))


def test_single_process_trains_and_checkpoints(tmp_path: Path) -> None:
    module = load_train_script()
    module.train(tiny_args(module, tmp_path))
    assert _checkpoints(tmp_path) == ["halo_vla_epoch1.pt", "halo_vla_epoch2.pt"]
    ckpt = torch.load(tmp_path / "halo_vla_epoch2.pt", weights_only=False)
    assert ckpt["epoch"] == 2 and ckpt["global_step"] == 16  # 8 batches x 2 epochs
    assert not any(k.startswith(("module.", "model.")) for k in ckpt["model_state_dict"])


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    module = load_train_script()
    module.train(tiny_args(module, tmp_path))
    module.train(tiny_args(module, tmp_path, "--epochs", "3", "--resume", str(tmp_path / "halo_vla_epoch2.pt")))
    ckpt = torch.load(tmp_path / "halo_vla_epoch3.pt", weights_only=False)
    assert ckpt["epoch"] == 3 and ckpt["global_step"] == 24


@pytest.mark.skipif(not HAS_WEBDATASET, reason="needs hale-blocks[webdataset]")
def test_webdataset_flag_converts_once_and_streams(tmp_path: Path) -> None:
    module = load_train_script()
    shards = tmp_path / "shards"
    args = tiny_args(module, tmp_path / "ckpt", "--webdataset", "--webdataset_root", str(shards),
                     "--webdataset_samples_per_shard", "4")
    module.train(args)
    (directory,) = [p for p in shards.iterdir() if p.is_dir()]
    tars = sorted(directory.glob("shard-*.tar"))
    assert len(tars) == 4 and (directory / "manifest.json").is_file()
    stamp = [t.stat().st_mtime_ns for t in tars]
    module.train(args)  # second run reuses the shards instead of rewriting them
    assert [t.stat().st_mtime_ns for t in tars] == stamp
    assert _checkpoints(tmp_path / "ckpt")[-1] == "halo_vla_epoch2.pt"


# ---------------------------------------------------------------------------- DDP (2 ranks)
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_rank(rank: int, world: int, port: int, tmp: str) -> None:
    """One torchrun-style rank: env vars set, process group created by DDPSession."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src" / "Halo_VLA")]
    module = load_train_script()
    built: list[TinyHaloVLM] = []

    class Recorded(TinyHaloVLM):
        def __init__(self, config) -> None:
            super().__init__(config)
            built.append(self)

    module.HaloVLM = Recorded
    module.train(tiny_args(module, Path(tmp) / "ckpt"))
    torch.save(built[0].state_dict(), Path(tmp) / f"rank{rank}.pt")


def test_ddp_two_ranks_stay_in_sync(tmp_path: Path) -> None:
    mp.spawn(_ddp_rank, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    a, b = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    assert all(torch.equal(a[k], b[k]) for k in a)  # gradients were all-reduced every step
    ckpt = torch.load(tmp_path / "ckpt" / "halo_vla_epoch2.pt", weights_only=False)
    assert ckpt["global_step"] == 8  # each rank saw half the batches: 4 per epoch
    assert all(torch.equal(ckpt["model_state_dict"][k], a[k]) for k in a)


# ---------------------------------------------------------------------------- Ray Train
@pytest.mark.skipif(not HAS_RAY, reason="needs hale-blocks[ray]")
def test_ray_two_workers_report_checkpoint(tmp_path: Path) -> None:
    module = load_train_script()
    args = tiny_args(module, tmp_path / "ckpt", "--ray", "--ray_num_workers", "2",
                     "--ray_storage_path", str(tmp_path / "ray"))
    # Workers import the entry by name (repo root is on their PYTHONPATH via RayCliOptions).
    result = module.launch_train_on_ray("tests.training_support:train_tiny", args)
    assert result.ok, result.error
    assert result.metrics["epoch"] == 2
    assert result.checkpoint_path is not None
    assert "halo_vla_epoch2.pt" in os.listdir(result.checkpoint_path)


def test_resolve_entry_accepts_script_path_and_module() -> None:
    from training.ray_launch import resolve_entry

    root = Path(__file__).resolve().parents[1]
    assert resolve_entry(f"{root / 'scripts' / 'train.py'}:train").__name__ == "train"
    assert resolve_entry("tests.training_support:train_tiny").__name__ == "train_tiny"
    with pytest.raises(ValueError):
        resolve_entry("no_function_given")


def test_validation_split_runs_the_evaluator_each_epoch(tmp_path: Path, monkeypatch) -> None:
    module = load_train_script()
    calls: list[dict] = []
    original = module.VLAEvaluator.run

    def recording_run(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(module.VLAEvaluator, "run", recording_run)
    module.train(tiny_args(module, tmp_path, "--val_fraction", "0.25"))
    assert len(calls) == 2  # one validation per epoch
    assert {"val/loss_total", "val/action_l1", "val/action_mse"} <= set(calls[-1])
    ckpt = torch.load(tmp_path / "halo_vla_epoch2.pt", weights_only=False)
    assert ckpt["global_step"] == 12  # 12 of 16 samples train: 6 batches x 2 epochs
