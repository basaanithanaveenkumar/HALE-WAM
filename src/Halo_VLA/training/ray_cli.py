"""``--ray*`` command-line flags of ``scripts/train.py`` -> HaleBlocks ``RayConfig``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

#: Repository root and the Halo_VLA source dir: scripts/train.py puts both on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[3]
IMPORT_ROOTS = (REPO_ROOT, REPO_ROOT / "src" / "Halo_VLA")


class RayCliOptions:
    """Adds the Ray Train flags to a parser and turns them into a ``RayConfig``.

    ``--ray`` alone launches ``train(args)`` on one local Ray worker; the other flags scale it
    out. Inside the workers the training loop runs with DDP (see :class:`DDPSession`).
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Ray Train (optional, needs hale-blocks[ray])")
        group.add_argument("--ray", action="store_true", help="Launch training on Ray Train workers")
        group.add_argument("--ray_address", default=None,
                           help="Ray cluster to join ('auto' or ray://host:10001); default: local Ray")
        group.add_argument("--ray_num_workers", type=int, default=1, help="Training processes (one per GPU)")
        group.add_argument("--ray_use_gpu", action="store_true", help="Give each worker a GPU")
        group.add_argument("--ray_cpus_per_worker", type=float, default=None, help="CPUs reserved per worker")
        group.add_argument("--ray_storage_path", default="ray_results",
                           help="Where Ray stores reported checkpoints (local, NFS or s3://)")
        group.add_argument("--ray_max_failures", type=int, default=0,
                           help="Restart the worker group this many times after a failure")

    @staticmethod
    def build_config(args: argparse.Namespace) -> Any:
        """The ``hale_core`` ``RayConfig`` for ``args`` (imports HaleBlocks lazily)."""
        try:
            from hale_core.config.sections.ray import RayConfig
        except ImportError as exc:  # pragma: no cover - depends on the environment
            from hale_core.exceptions import MissingDependencyError

            raise MissingDependencyError(
                "hale_core.config.sections.ray", "--ray needs HaleBlocks with Ray: pip install -e '.[ray]'"
            ) from exc
        resources = {}
        if args.ray_cpus_per_worker is not None:
            resources["CPU"] = args.ray_cpus_per_worker
        if args.ray_use_gpu:
            resources["GPU"] = 1
        # Ray workers are fresh processes: give them the same import roots as the script
        # (works locally and on clusters where the repo lives at the same path, e.g. NFS).
        pythonpath = [str(p) for p in IMPORT_ROOTS]
        if os.environ.get("PYTHONPATH"):
            pythonpath.append(os.environ["PYTHONPATH"])
        return RayConfig(
            enabled=True,
            runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(pythonpath)}},
            address=args.ray_address,
            num_workers=args.ray_num_workers,
            use_gpu=args.ray_use_gpu,
            resources_per_worker=resources,
            storage_path=args.ray_storage_path,
            max_failures=args.ray_max_failures,
        )
