"""Tensors or batches passed to a Halo-VLA module that it cannot use."""

from __future__ import annotations

from exceptions.halo_vla_error import HaloVLAError


class ModelInputError(HaloVLAError, ValueError):
    """A forward/loss call received inputs with the wrong shape or missing required entries
    (e.g. no ``<halo_action>`` token in ``input_ids``).
    """
