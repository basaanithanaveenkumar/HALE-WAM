"""Invalid Halo-VLA model, dual-system, run or CLI configuration."""

from __future__ import annotations

from hale_core.exceptions import ConfigError

from exceptions.halo_vla_error import HaloVLAError


class HaloVLAConfigError(HaloVLAError, ConfigError):
    """A configuration value is invalid, inconsistent or missing.

    Also a :class:`hale_core.exceptions.ConfigError` and therefore a :class:`ValueError`, so pydantic
    validators wrap it into a ``ValidationError`` and existing ``except ValueError`` sites still match.
    """
