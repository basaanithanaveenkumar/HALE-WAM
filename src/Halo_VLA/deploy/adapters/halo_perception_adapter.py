"""(a) Single-system perception graph: observation + prompt -> action conditioning."""

from __future__ import annotations

from hale_core.deploy.core.io_signature import IOSignature
from torch import nn

from Halo_VLA.deploy.adapters.observation_prompt_adapter import ObservationPromptAdapter
from Halo_VLA.deploy.graphs.perception_graph import PerceptionGraph
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class HaloPerceptionAdapter(ObservationPromptAdapter):
    """Output ``action_context [B, num_action_tokens, emb_dim]`` (the flow head's ``cond``).

    The heavy part of the policy (vision encoder + VLM decoder); runs once per camera frame.
    """

    requires_dual = False

    def default_signature(self) -> IOSignature:
        """Observation and prompt in; the action conditioning out."""
        return self.observation_signature(["action_context"])

    def graph(self, model: nn.Module) -> nn.Module:
        """Vision encoder and decoder up to the action-token states."""
        return PerceptionGraph(model, self.num_action_tokens)
