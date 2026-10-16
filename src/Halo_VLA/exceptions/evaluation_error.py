"""Evaluation cannot run on the given model/data combination."""

from __future__ import annotations

from exceptions.halo_vla_error import HaloVLAError


class EvaluationError(HaloVLAError, ValueError):
    """The validation data is empty or incompatible with the evaluator (e.g. action width mismatch).
    """
