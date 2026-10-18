"""(c) Single-system policy in one call (perception + unrolled Euler sampler)."""

from __future__ import annotations

from typing import Any

from torch import nn

from hale_core.deploy.core.io_signature import IOSignature
from Halo_VLA.deploy.adapters.observation_prompt_adapter import ObservationPromptAdapter
from Halo_VLA.deploy.graphs.flow_policy_graph import FlowPolicyGraph
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class HaloFlowPolicyAdapter(ObservationPromptAdapter):
    """Inputs: observation + prompt + ``noise [B, K, chunk*action_dim]``; output ``actions
    [B, K, chunk, action_dim]``. Reference graph for parity and for synchronous deployments.

    Args:
        num_steps: Euler steps unrolled into the graph (default ``flow_num_ode_steps``).
    """

    requires_dual = False

    def __init__(self, num_steps: int | None = None, **base: Any) -> None:
        super().__init__(**base)
        self.num_steps = num_steps

    def default_signature(self) -> IOSignature:
        """Observation, prompt and noise in; the sampled action chunks out."""
        cfg = self.model_config()
        noise = {"name": "noise", "shape": ["batch", self.num_action_tokens, cfg.action_chunk_size * cfg.action_dim],
                 "dtype": "float32", "value_range": [-2.0, 2.0]}
        return self.observation_signature(["actions"], [noise])

    def graph(self, model: nn.Module) -> nn.Module:
        """Perception followed by the unrolled Euler sampler."""
        return FlowPolicyGraph(model, self.num_action_tokens, self.num_steps)
