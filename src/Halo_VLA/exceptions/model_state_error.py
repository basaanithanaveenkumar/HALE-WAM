"""A model is not in a state that allows the requested operation."""

from __future__ import annotations

from exceptions.halo_vla_error import HaloVLAError


class ModelStateError(HaloVLAError, RuntimeError):
    """The model was built in a way that cannot serve the call (e.g. no decoder layers to hook,
    a world model that produced no prediction).
    """
