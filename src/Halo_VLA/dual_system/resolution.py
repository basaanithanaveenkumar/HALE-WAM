"""Turn a preset name plus overrides into a validated :class:`DualSystemConfig`.

Example:
    >>> from dual_system.resolution import resolve_dual_system_config, parse_overrides
    >>> cfg = resolve_dual_system_config("lcb", parse_overrides(["async_steps=4"]))
    >>> cfg.async_steps
    4
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from hale_core.exceptions import RegistryError

from dual_system.config import DualSystemConfig
from dual_system.presets import DUAL_SYSTEM_PRESETS
from exceptions import HaloVLAConfigError

#: ``dual_system`` value that disables the dual system.
NONE = "none"


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse ``["latent=pooled", "async_steps=4"]`` into typed overrides.

    Args:
        items: ``KEY=VALUE`` strings; ``None`` or an empty list gives no overrides.

    Returns:
        ``{field: value}`` with each value converted to the dataclass field's type
        (``none`` / ``null`` become ``None``).

    Raises:
        HaloVLAConfigError: for an item without ``=`` or with an unknown key.
    """
    types = {f.name: f.type for f in fields(DualSystemConfig)}
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or key not in types:
            raise HaloVLAConfigError(
                f"bad dual-system option {item!r}; use KEY=VALUE with KEY in {sorted(types)}"
            )
        kind = str(types[key])
        if raw.lower() in ("none", "null"):
            value: Any = None
        elif "bool" in kind:
            value = raw.lower() in ("1", "true", "yes")
        elif "int" in kind:
            value = int(raw)
        elif "float" in kind:
            value = float(raw)
        else:
            value = raw
        out[key] = value
    return out


def resolve_dual_system_config(
    name: str | None, overrides: dict[str, Any] | None = None
) -> DualSystemConfig | None:
    """The preset ``name`` (or ``None`` for ``none``) with ``overrides`` applied.

    ``name="custom"`` starts from the dataclass defaults, so any combination can be built from
    overrides alone.

    Args:
        name: ``none`` / ``None``, ``custom`` or a registered preset name.
        overrides: ``DualSystemConfig`` fields to replace.

    Returns:
        The validated config, or ``None`` when the dual system is disabled.

    Raises:
        HaloVLAConfigError: for an unknown preset or override key, or overrides with ``none``.
    """
    if name is None or name == NONE:
        if overrides:
            raise HaloVLAConfigError("dual_system options given but dual_system is 'none'")
        return None
    if name == "custom":
        base = DualSystemConfig()
    else:
        try:
            base = DUAL_SYSTEM_PRESETS.get(name)
        except RegistryError as err:
            raise HaloVLAConfigError(
                f"unknown dual_system {name!r}; choose none, custom or one of {sorted(DUAL_SYSTEM_PRESETS.keys())}"
            ) from err
    overrides = dict(overrides or {})
    unknown = set(overrides) - {f.name for f in fields(DualSystemConfig)}
    if unknown:
        raise HaloVLAConfigError(f"unknown dual_system options: {sorted(unknown)}")
    return replace(base, **overrides)


__all__ = ["NONE", "parse_overrides", "resolve_dual_system_config"]
