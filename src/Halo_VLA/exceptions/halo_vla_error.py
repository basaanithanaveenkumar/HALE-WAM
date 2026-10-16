"""Root of the Halo-VLA exception hierarchy."""

from __future__ import annotations

from hale_core.exceptions import HaleCoreError


class HaloVLAError(HaleCoreError):
    """Base class of every error Halo-VLA raises on purpose.

    It derives from :class:`hale_core.exceptions.HaleCoreError`, so a launcher can catch every
    expected Hale* failure with one ``except`` and let programming errors propagate.
    """
