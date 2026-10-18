"""Dual-system VLA: a slow vision-language System 2 guides a fast action System 1.

Built from first principles and configurable into the published designs surveyed by OpenHelix
(Cui et al., 2025, arXiv:2505.03912): ``basic``, ``lcb``, ``dp_vla``, ``hirt``, ``robodual``,
``dexvla``, ``openhelix`` - or ``custom`` / ``none``. See ``docs/dual_system.md``.

* :class:`DualSystemConfig`, :data:`DUAL_SYSTEM_PRESETS` (registry; :data:`PRESETS` is its
  mapping), :func:`resolve_dual_system_config`
* :class:`System2` - Halo-VLA's VLM trunk with learnable ``<ACT>`` queries
* latent extractors (:data:`LATENT_EXTRACTORS`) + :class:`LatentBridge` - what crosses from
  System 2 to System 1
* :class:`System1Policy` - fast policy; regression / diffusion / flow matching are strategies in
  :data:`SYSTEM1_POLICIES`
* :class:`DualSystemVLA` - the assembled model and its losses
* :class:`DualSystemController` - asynchronous two-frequency inference
"""

from dual_system.config import DualSystemConfig
from dual_system.controller import DualSystemController
from dual_system.latent import LATENT_EXTRACTORS, LatentBridge, build_latent_extractor
from dual_system.model import DualSystemLosses, DualSystemVLA
from dual_system.presets import DUAL_SYSTEM_PRESETS, PRESETS, register_preset
from dual_system.resolution import parse_overrides, resolve_dual_system_config
from dual_system.system1 import SYSTEM1_POLICIES, System1Policy
from dual_system.system2 import System2

__all__ = [
    "DUAL_SYSTEM_PRESETS",
    "LATENT_EXTRACTORS",
    "PRESETS",
    "SYSTEM1_POLICIES",
    "DualSystemConfig",
    "DualSystemController",
    "DualSystemLosses",
    "DualSystemVLA",
    "LatentBridge",
    "System1Policy",
    "System2",
    "build_latent_extractor",
    "parse_overrides",
    "register_preset",
    "resolve_dual_system_config",
]
