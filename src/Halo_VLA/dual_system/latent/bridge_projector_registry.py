"""Registry of latent projectors for :class:`LatentBridge`, keyed by ``DualSystemConfig.projector``."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from hale_core.registry import NamedRegistry

#: ``projector`` name -> ``(in_dim, out_dim) -> nn.Module``.
BRIDGE_PROJECTORS = NamedRegistry("bridge projector")


@BRIDGE_PROJECTORS.register_module(name="linear")
def _linear_projector(in_dim: int, out_dim: int) -> nn.Module:
    """A single linear map (OpenHelix)."""
    return nn.Linear(in_dim, out_dim)


@BRIDGE_PROJECTORS.register_module(name="mlp")
def _mlp_projector(in_dim: int, out_dim: int) -> nn.Module:
    """A 2-layer MLP with a GELU in between."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))


ProjectorFactory = Callable[[int, int], nn.Module]

__all__ = ["BRIDGE_PROJECTORS", "ProjectorFactory"]
