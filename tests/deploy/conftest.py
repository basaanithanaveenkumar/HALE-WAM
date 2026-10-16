"""Fixtures of the deployment tests (helpers live in :mod:`tests.deploy.support`)."""

from __future__ import annotations

import pytest
import torch

from tests.deploy.support import TINY_MODEL

pytest.importorskip("hale_core.deploy", reason="HaleBlocks (hale_core.deploy) is required")

import Halo_VLA.deploy  # noqa: E402,F401 - registers the adapters


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "integration: multi-module workflows (real exports / runtimes)")


@pytest.fixture
def tiny_vlm():
    from config import HaloVLMConfig
    from models.halo_vla import HaloVLM

    torch.manual_seed(0)
    return HaloVLM(HaloVLMConfig(**TINY_MODEL)).eval()
