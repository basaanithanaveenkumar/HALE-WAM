"""A training entry point (``script.py:fn``) cannot be imported."""

from __future__ import annotations

from exceptions.halo_vla_error import HaloVLAError


class TrainingEntryError(HaloVLAError, ImportError):
    """The ``--entry`` given to the Ray launcher does not name an importable training function.
    """
