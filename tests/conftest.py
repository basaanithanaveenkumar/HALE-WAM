"""Shared pytest configuration: import roots, deterministic seeds, unit / integration markers.

Layout: ``tests/unit`` (fast, isolated, no network / GPU), ``tests/integration`` (end-to-end flows
on real configs), ``tests/support`` (fakes and helpers shared by both).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
# The sources are not one installed package: the repo root (``config``, ``dataloader``) and
# ``src/Halo_VLA`` (``models``, ``dual_system``) are import roots, ``src`` holds ``Halo_VLA.deploy``.
for _path in (ROOT, ROOT / "src" / "Halo_VLA", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed Python and torch before every test so results do not depend on execution order."""
    random.seed(0)
    torch.manual_seed(0)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests by directory: everything under ``tests/unit`` is ``unit``, ``tests/integration`` is ``integration``."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "unit" in parts:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def tiny_vlm():
    """A 2-layer, 64-wide ``HaloVLM`` in eval mode (the deployment CI model)."""
    from config import HaloVLMConfig
    from models.halo_vla import HaloVLM
    from tests.support.deploy_support import TINY_MODEL

    torch.manual_seed(0)
    return HaloVLM(HaloVLMConfig(**TINY_MODEL)).eval()
