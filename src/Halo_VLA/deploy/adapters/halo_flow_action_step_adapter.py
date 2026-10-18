"""(b) Single-system action-head step: one velocity evaluation of the flow-matching decoder."""

from __future__ import annotations

from typing import Any

import torch
from hale_core.deploy.core.io_signature import IOSignature
from torch import nn

from Halo_VLA.deploy.adapters.halo_adapter_base import HaloAdapterBase
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class HaloFlowActionStepAdapter(HaloAdapterBase):
    """``noisy_actions [N, chunk*action_dim], t [N], context [N, emb_dim] -> velocity``.

    ``N = batch x action tokens``. The host owns the integrator (``x += v * dt`` for
    ``flow_num_ode_steps`` steps, ``t = step * dt``): it can change step count, use a better
    solver or stop early without re-exporting - the step is a ~100 kFLOP MLP, ideal for an NPU.
    """

    requires_dual = False

    def __init__(self, max_rows: int | None = None, **base: Any) -> None:
        super().__init__(**base)
        self.max_rows = int(max_rows or self.max_batch)

    def default_signature(self) -> IOSignature:
        """Noisy action rows, time and context in; the velocity out."""
        cfg = self.model_config()
        flat = cfg.action_chunk_size * cfg.action_dim
        return self.build_signature(
            {
                "inputs": [
                    {
                        "name": "noisy_actions",
                        "shape": ["rows", flat],
                        "dtype": "float32",
                        "value_range": [-2.0, 2.0],
                    },
                    {"name": "t", "shape": ["rows"], "dtype": "float32", "value_range": [0.0, 1.0]},
                    {
                        "name": "context",
                        "shape": ["rows", cfg.emb_dim],
                        "dtype": "float32",
                        "value_range": [-1.0, 1.0],
                    },
                ],
                "outputs": ["velocity"],
                "dims": {"rows": [1, 1, self.max_rows]},
            }
        )

    def graph(self, model: nn.Module) -> nn.Module:
        """The flow-matching decoder."""
        return model.flow_decoder

    def forward_for_export(self, graph: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the velocity field once.

        Args:
            graph: the flow decoder.
            inputs: ``noisy_actions``, ``t``, ``context``.

        Returns:
            The velocity ``[N, chunk * action_dim]``.
        """
        return graph(inputs["noisy_actions"], inputs["t"], inputs["context"])
