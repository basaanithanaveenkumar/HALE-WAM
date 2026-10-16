"""Halo-VLA's scope in the deployment registries (mmdet-style child registry).

``ADAPTERS`` (HaleBlocks) is the parent: configs can say ``type: HaloPerceptionAdapter`` (found
here) or ``type: halo_vla.HaloPerceptionAdapter`` (scope-qualified from anywhere), while every
built-in adapter stays reachable through the parent.
"""

from __future__ import annotations

from hale_core.deploy.registry import ADAPTERS, Registry

HALO_VLA_ADAPTERS = Registry("adapter", parent=ADAPTERS, scope="halo_vla")

__all__ = ["HALO_VLA_ADAPTERS"]
