"""(c') Dual-system System 1 in one call: latent + observation (+ noise) -> action chunk."""

from __future__ import annotations

from typing import Any

from hale_core.deploy.core.io_signature import IOSignature
from torch import nn

from Halo_VLA.deploy.adapters.dual_system1_adapter_base import DualSystem1AdapterBase
from Halo_VLA.deploy.graphs.system1_policy_graph import System1PolicyGraph
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class DualSystem1PolicyAdapter(DualSystem1AdapterBase):
    """The fast control-rate graph of a dual-system robot: condition + full sampler unrolled.

    Args:
        steps: sampler steps unrolled into the graph (default ``sample_steps``).
    """

    def __init__(self, steps: int | None = None, **base: Any) -> None:
        super().__init__(**base)
        self.steps = steps

    def default_signature(self) -> IOSignature:
        """Latent, observation and (for sampling policies) noise in; the action chunk out."""
        from dual_system.system1.system1_registry import (
            SYSTEM1_POLICIES,  # lazy: import roots are set up by Halo_VLA.deploy
        )

        cfg, dual = self.model_config(), self.dual_config()
        inputs = self.latent_inputs()
        if SYSTEM1_POLICIES.get(dual.policy).needs_noise:
            inputs.append(
                {
                    "name": "noise",
                    "shape": ["batch", cfg.action_chunk_size, cfg.action_dim],
                    "dtype": "float32",
                    "value_range": [-2.0, 2.0],
                }
            )
        return self.build_signature(
            {
                "inputs": inputs,
                "outputs": ["actions"],
                "dims": {"batch": self.batch_dim(), "latent_len": self.latent_dim()},
            }
        )

    def graph(self, model: nn.Module) -> nn.Module:
        """The condition-plus-unrolled-sampler graph."""
        return System1PolicyGraph(model, self.steps)
