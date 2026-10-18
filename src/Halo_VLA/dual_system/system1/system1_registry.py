"""Registry of System 1 action-policy strategies, keyed by ``DualSystemConfig.policy``."""

from __future__ import annotations

from hale_core.registry import NamedRegistry

#: ``policy`` name -> :class:`~dual_system.system1.action_policy_strategy.ActionPolicyStrategy` subclass.
SYSTEM1_POLICIES = NamedRegistry("system 1 policy")

__all__ = ["SYSTEM1_POLICIES"]
