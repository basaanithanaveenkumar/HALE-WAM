"""Run ``scripts/train.py``'s ``train(args)`` on Ray Train workers."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from training.ray_cli import RayCliOptions
from exceptions import HaloVLAConfigError, TrainingEntryError

#: Module name under which a training *script* (``path/to/train.py``) is imported in a worker.
_SCRIPT_MODULE = "halo_vla_ray_entry"


def resolve_entry(entry: str) -> Callable[[argparse.Namespace], Any]:
    """``"path/to/script.py:function"`` or ``"package.module:function"`` -> the function.

    Workers resolve the training function from this reference instead of receiving a pickled
    copy: pickling a script's ``train`` by value would also pickle its module globals (loggers
    with open streams, TensorBoard writers ...), which is fragile or impossible.
    """
    target, sep, name = entry.rpartition(":")
    if not sep or not target or not name:
        raise HaloVLAConfigError(f"entry must look like 'path/to/script.py:train' or 'pkg.mod:train', got {entry!r}")
    if target.endswith(".py"):
        module = sys.modules.get(_SCRIPT_MODULE)
        if module is None or getattr(module, "__file__", None) != str(Path(target).resolve()):
            spec = importlib.util.spec_from_file_location(_SCRIPT_MODULE, Path(target).resolve())
            if spec is None or spec.loader is None:
                raise TrainingEntryError(f"cannot import training script {target}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_SCRIPT_MODULE] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, name)


def _ray_worker(loop_config: dict[str, Any]) -> None:
    """Executed in every Ray worker: the normal training function (DDP via DDPSession)."""
    resolve_entry(loop_config["entry"])(loop_config["args"])


def launch_train_on_ray(entry: str, args: argparse.Namespace) -> Any:
    """Launch the training function ``entry`` with ``args`` on ``--ray_num_workers`` workers.

    Args:
        entry: ``"path/to/train.py:train"`` (what ``scripts/train.py`` passes) or an importable
            ``"package.module:function"``. Workers must see the same path / package (same
            machine, a shared filesystem, or ``runtime_env`` ``working_dir``).
        args: the parsed command line (a plain ``Namespace``, shipped to every worker).

    Returns:
        HaleBlocks' ``RayRunResult``: the last reported metrics and checkpoint directory.
        Inside a worker, ``DDPSession.start`` finds the process group Ray created, so the same
        training function runs data-parallel; checkpoints reported from rank 0 end up under
        ``--ray_storage_path``.
    """
    from hale_core.distributed import RayLauncher

    resolve_entry(entry)  # fail fast on the driver if the reference is wrong
    return RayLauncher(RayCliOptions.build_config(args)).launch(
        _ray_worker, {"entry": entry, "args": args}
    )
