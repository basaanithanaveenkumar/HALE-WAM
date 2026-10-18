"""(b) Dual-system System 1 action-head step: one transformer evaluation of the sampler."""

from __future__ import annotations

import torch
from torch import nn

from hale_core.deploy.core.dim_spec import DimSpec
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import ConfigError
from Halo_VLA.deploy.adapters.dual_system1_adapter_base import DualSystem1AdapterBase
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class DualSystem1ActionStepAdapter(DualSystem1AdapterBase):
    """``condition, condition_mask, noisy_actions, timestep -> prediction``.

    The host runs the sampler: flow matching ``x += head(..., t * 1000) / steps``; diffusion
    DDIM with v-prediction over ``alpha_bar`` (cosine schedule, see ``System1Policy.sample``).
    ``timestep`` is passed already scaled (``t * 1000`` or the integer noise level as float).
    Regression policies have no step: export :class:`DualSystem1PolicyAdapter` instead.
    """

    def default_signature(self) -> IOSignature:
        """Condition tokens, noisy actions and timestep in; the head's prediction out.

        Raises:
            ConfigError: for a regression policy, which has no sampler step.
        """
        from dual_system.system1.system1_registry import SYSTEM1_POLICIES  # lazy: import roots are set up by Halo_VLA.deploy

        cfg, dual = self.model_config(), self.dual_config()
        if not SYSTEM1_POLICIES.get(dual.policy).has_sampler_step:
            raise ConfigError("regression System 1 has no sampler step; use DualSystem1PolicyAdapter")
        latent = DimSpec.from_config("latent_len", self.latent_dim())
        extra = self.observation_tokens() + 1  # observation tokens + proprioception token
        context = [latent.min + extra, latent.opt + extra, latent.max + extra]
        return self.build_signature({
            "inputs": [
                {"name": "condition", "shape": ["batch", "context", dual.system1_dim], "dtype": "float32",
                 "value_range": [-1.0, 1.0]},
                {"name": "condition_mask", "shape": ["batch", "context"], "dtype": "int64", "value_range": [1, 2]},
                {"name": "noisy_actions", "shape": ["batch", cfg.action_chunk_size, cfg.action_dim],
                 "dtype": "float32", "value_range": [-2.0, 2.0]},
                {"name": "timestep", "shape": ["batch"], "dtype": "float32", "value_range": [0.0, 1000.0]},
            ],
            "outputs": ["prediction"],
            "dims": {"batch": self.batch_dim(), "context": context},
        })

    def graph(self, model: nn.Module) -> nn.Module:
        """The System 1 action transformer."""
        return model.system1.head

    def forward_for_export(self, graph: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Call the transformer with the named inputs.

        Args:
            graph: the System 1 action transformer.
            inputs: ``condition``, ``condition_mask``, ``noisy_actions``, ``timestep``.

        Returns:
            The velocity (flow matching) or v-prediction (diffusion) ``[B, chunk, action_dim]``.
        """
        return graph(inputs["condition"], inputs["condition_mask"], inputs["noisy_actions"], inputs["timestep"])
