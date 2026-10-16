"""A checkpoint file to resume or evaluate from does not exist."""

from __future__ import annotations

from exceptions.halo_vla_error import HaloVLAError


class CheckpointNotFoundError(HaloVLAError, FileNotFoundError):
    """The checkpoint path given on the command line or in a config does not exist.
    """
