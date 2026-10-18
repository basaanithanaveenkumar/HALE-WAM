"""A dataset or data stage cannot produce samples."""

from __future__ import annotations

from exceptions.halo_vla_error import HaloVLAError


class DatasetError(HaloVLAError, RuntimeError):
    """No usable dataset was configured or a stage resolved to zero sources."""
