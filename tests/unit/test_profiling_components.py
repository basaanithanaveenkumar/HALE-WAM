"""The profiling stub registry: every component builds and runs a forward pass."""

from __future__ import annotations

import pytest
import torch

from profiling import PROFILE_COMPONENTS, ProfileComponentBuilder


@pytest.mark.parametrize("component", list(PROFILE_COMPONENTS))
def test_component_builds_and_runs(component: str) -> None:
    builder = ProfileComponentBuilder(batch_size=1, emb_dim=32, n_layers=1, n_heads=2, mlp_dim=64)
    model, dummy = builder.build(component)
    with torch.no_grad():
        out = model(*dummy) if isinstance(dummy, tuple) else model(dummy)
    assert torch.isfinite(out).all()


def test_unknown_component_raises_registry_error() -> None:
    from hale_core.exceptions import RegistryError

    builder = ProfileComponentBuilder(batch_size=1, emb_dim=32, n_layers=1, n_heads=2, mlp_dim=64)
    with pytest.raises(RegistryError, match="unknown profile component"):
        builder.build("nope")
