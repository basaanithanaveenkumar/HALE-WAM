"""Build the Halo-VLA network a deployment adapter exports from (HaloVLM or DualSystemVLA)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from hale_core.deploy.exceptions import ConfigError
from loguru import logger
from torch import nn

from exceptions import HaloVLAConfigError

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig


class HaloModelFactory:
    """``HaloVLMConfig`` fields + optional dual-system preset -> the exact training-time module.

    The module tree (and therefore the state-dict keys) matches ``scripts/train.py``: a bare
    ``HaloVLM`` when ``dual_system`` is ``none`` and ``DualSystemVLA(HaloVLM, preset)`` otherwise,
    so training checkpoints (``model_state_dict``) load without key remapping.

    Args:
        model: ``HaloVLMConfig`` keyword arguments (the architecture you trained).
        dual_system: preset name (``basic``, ``lcb``, ``openhelix`` ...), ``custom`` or ``none``.
        dual_system_options: overrides of the dual-system preset.
        seed: initialisation seed (matters only without a checkpoint).
    """

    def __init__(
        self,
        model: Mapping[str, Any] | None = None,
        dual_system: str | None = "none",
        dual_system_options: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.model_kwargs = dict(model or {})
        self.dual_system = dual_system or "none"
        self.dual_system_options = dict(dual_system_options or {})
        self.seed = seed

    @property
    def is_dual(self) -> bool:
        """Whether a dual-system preset is configured."""
        return self.dual_system != "none"

    def config(self) -> HaloVLMConfig:
        """The validated ``HaloVLMConfig``."""
        from config import HaloVLMConfig

        try:
            return HaloVLMConfig(**self.model_kwargs)
        except TypeError as exc:
            raise ConfigError(f"invalid HaloVLMConfig fields in adapter.model: {exc}") from exc

    def build(self) -> nn.Module:
        """Build the network, seeded for reproducible initialisation.

        Returns:
            A ``HaloVLM`` or, with a dual-system preset, a ``DualSystemVLA``.

        Raises:
            ConfigError: for invalid dual-system settings.
        """
        from dual_system import DualSystemVLA, resolve_dual_system_config
        from models.halo_vla import HaloVLM

        torch.manual_seed(self.seed)
        vlm = HaloVLM(self.config())
        if not self.is_dual:
            return vlm
        try:
            dual_cfg = resolve_dual_system_config(self.dual_system, self.dual_system_options)
        except HaloVLAConfigError as exc:
            raise ConfigError(f"invalid dual-system settings: {exc}") from exc
        logger.info(
            "dual-system VLA preset={} policy={} latent={}",
            dual_cfg.name,
            dual_cfg.policy,
            dual_cfg.latent,
        )
        return DualSystemVLA(vlm, dual_cfg)
