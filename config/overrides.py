"""Set ``HaloVLMConfig`` fields by name: YAML architecture keys and ``--model_option KEY=VALUE``."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from config.model_config import HaloVLMConfig

_FIELDS = {f.name: str(f.type) for f in dataclasses.fields(HaloVLMConfig)}


def _parse(kind: str, raw: str) -> Any:
    """``raw`` typed by the dataclass annotation ``kind`` (``"int | None"``, ``"tuple[int, ...]"`` ...)."""
    if raw.lower() in ("none", "null"):
        return None
    if kind.startswith(("tuple", "list")):
        items = [x.strip() for x in raw.strip("[]()").split(",") if x.strip()]
        item_kind = "int" if "int" in kind else "str"
        return [_parse(item_kind, x) for x in items]
    if kind.startswith("dict"):
        return json.loads(raw)
    if kind.startswith("bool"):
        if raw.lower() not in ("1", "0", "true", "false", "yes", "no"):
            raise ValueError(f"expected a boolean, got {raw!r}")
        return raw.lower() in ("1", "true", "yes")
    if kind.startswith("int"):
        return int(raw)
    if kind.startswith("float"):
        return float(raw)
    return raw


def parse_model_options(items: list[str] | None) -> dict[str, Any]:
    """``["emb_dim=256", "use_moe=false"]`` -> typed ``{field: value}`` for :class:`HaloVLMConfig`."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or key not in _FIELDS:
            raise ValueError(f"bad model option {item!r}; use KEY=VALUE with KEY a HaloVLMConfig field")
        out[key] = _parse(_FIELDS[key], raw)
    return out


def apply_model_overrides(config: HaloVLMConfig, overrides: dict[str, Any]) -> HaloVLMConfig:
    """Set ``overrides`` on ``config`` (lists become tuples for tuple fields) and re-validate it."""
    unknown = set(overrides) - set(_FIELDS)
    if unknown:
        raise ValueError(f"unknown HaloVLMConfig fields: {sorted(unknown)}")
    for key, value in overrides.items():
        if _FIELDS[key].startswith("tuple") and value is not None:
            value = tuple(value)
        setattr(config, key, value)
    config.__post_init__()
    return config
