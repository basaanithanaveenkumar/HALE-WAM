"""Make Halo-VLA's flat import roots available to the deployment package.

The Halo-VLA sources are not one installed package: ``scripts/train.py`` and the tests put the
repository root (``config``, ``dataloader``) and ``src/Halo_VLA`` (``models``, ``dual_system``)
on ``sys.path`` and import ``from models.halo_vla import HaloVLM``. The deployment package lives
at ``src/Halo_VLA/deploy`` (importable as ``Halo_VLA.deploy`` with ``src`` on the path) and needs
the same roots before it imports any model code.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


class ImportRoots:
    """Idempotently prepend the repository root and ``src/Halo_VLA`` to ``sys.path``.

    Attributes:
        repo_root: the HALE-WAM checkout (contains ``config/`` and ``src/``).
        source_root: ``src/Halo_VLA`` (contains ``models/`` and ``dual_system/``).
    """

    def __init__(self, anchor: Path | None = None) -> None:
        here = (anchor or Path(__file__)).resolve()
        self.source_root = here.parents[1]  # .../src/Halo_VLA
        self.repo_root = here.parents[3]    # .../HALE-WAM

    def ensure(self) -> list[str]:
        """Add the roots that are missing; returns the paths that were added."""
        added = []
        for path in (str(self.repo_root), str(self.source_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
                added.append(path)
        if added:
            logger.debug("Halo-VLA import roots added to sys.path: {}", added)
        return added
