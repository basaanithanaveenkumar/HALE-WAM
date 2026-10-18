"""(a') Dual-system System 1 observer: latent + current frame + proprioception -> condition tokens."""

from __future__ import annotations

import torch
from hale_core.deploy.core.io_signature import IOSignature
from torch import nn

from Halo_VLA.deploy.adapters.dual_system1_adapter_base import DualSystem1AdapterBase
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class DualSystem1ObserverAdapter(DualSystem1AdapterBase):
    """``System1Policy.condition``: outputs ``condition [B, C, system1_dim]``, ``condition_mask``.

    Runs once per control step (fast conv encoder + optional Perceiver resampler); its output is
    reused by every sampler step of :class:`DualSystem1ActionStepAdapter`.
    """

    def default_signature(self) -> IOSignature:
        """Latent, latent mask, image and state in; condition tokens and mask out."""
        return self.build_signature(
            {
                "inputs": self.latent_inputs(),
                "outputs": ["condition", "condition_mask"],
                "dims": {"batch": self.batch_dim(), "latent_len": self.latent_dim()},
            }
        )

    def graph(self, model: nn.Module) -> nn.Module:
        """The whole System 1 policy (only ``condition`` is called)."""
        return model.system1

    def forward_for_export(
        self, graph: nn.Module, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``System1Policy.condition``.

        Args:
            graph: the System 1 policy.
            inputs: ``latent``, ``latent_mask``, ``image``, ``state``.

        Returns:
            ``(condition, condition_mask)``.
        """
        return graph.condition(
            inputs["latent"], inputs["latent_mask"], inputs["image"], inputs["state"]
        )
