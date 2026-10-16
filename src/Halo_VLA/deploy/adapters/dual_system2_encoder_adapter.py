"""(a) Dual-system System 2: observation + instruction -> latent for System 1 (slow, async)."""

from __future__ import annotations

from torch import nn

from hale_core.deploy.core.io_signature import IOSignature
from Halo_VLA.deploy.adapters.observation_prompt_adapter import ObservationPromptAdapter
from Halo_VLA.deploy.graphs.system2_latent_graph import System2LatentGraph
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class DualSystem2EncoderAdapter(ObservationPromptAdapter):
    """Outputs ``latent [B, K, system1_dim]`` and ``latent_mask [B, K]``.

    Runs every ``async_steps`` control steps (``DualSystemController``); ``K`` depends on the
    preset's latent type (1 for ``act_token`` / ``pooled``, the prompt length for
    ``last_hidden`` ...).
    """

    requires_dual = True

    def default_signature(self) -> IOSignature:
        return self.observation_signature(["latent", "latent_mask"])

    def graph(self, model: nn.Module) -> nn.Module:
        return System2LatentGraph(model)
