"""(c) Whole dual-system policy in one call (System 2 + System 1), reference / synchronous use."""

from __future__ import annotations

from typing import Any

from torch import nn

from hale_core.deploy.core.io_signature import IOSignature
from Halo_VLA.deploy.adapters.observation_prompt_adapter import ObservationPromptAdapter
from Halo_VLA.deploy.graphs.dual_system_policy_graph import DualSystemPolicyGraph
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class DualSystemPolicyAdapter(ObservationPromptAdapter):
    """Observation + prompt (+ noise) -> ``actions [B, chunk, action_dim]``.

    Args:
        steps: System 1 sampler steps (default ``sample_steps``).
    """

    requires_dual = True

    def __init__(self, steps: int | None = None, **base: Any) -> None:
        super().__init__(**base)
        self.steps = steps

    def default_signature(self) -> IOSignature:
        cfg = self.model_config()
        extra = []
        if self.dual_config().policy != "regression":
            extra.append({"name": "noise", "shape": ["batch", cfg.action_chunk_size, cfg.action_dim],
                          "dtype": "float32", "value_range": [-2.0, 2.0]})
        return self.observation_signature(["actions"], extra)

    def graph(self, model: nn.Module) -> nn.Module:
        return DualSystemPolicyGraph(model, self.steps)
