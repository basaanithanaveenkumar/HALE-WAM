"""A dataset root directory does not exist."""

from __future__ import annotations

from exceptions.halo_vla_error import HaloVLAError


class DatasetNotFoundError(HaloVLAError, FileNotFoundError):
    """``data_root`` (or a dataset sub-path) does not exist or is not a directory."""
