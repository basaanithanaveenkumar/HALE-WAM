"""Deployment tests need HaleBlocks' deploy framework and the registered Halo-VLA adapters."""

from __future__ import annotations

import pytest

pytest.importorskip("hale_core.deploy", reason="HaleBlocks (hale_core.deploy) is required")

import Halo_VLA.deploy  # noqa: E402,F401 - registers the adapters
