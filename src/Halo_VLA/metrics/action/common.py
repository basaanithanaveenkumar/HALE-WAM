"""Helpers shared by the ``action`` package."""

from __future__ import annotations

from typing import Optional

from torch import Tensor


def _apply_mask(t: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Flatten a [B, T, D] tensor, applying a [B, T] boolean mask if given."""
    if mask is None:
        return t.reshape(-1, t.shape[-1])
    # mask: True where the timestep is valid
    return t[mask.bool()]  # [N_valid, D]
