"""Single-system policy in one call: perception + unrolled Euler integration of the flow head."""

from __future__ import annotations

import torch
from torch import nn

from Halo_VLA.deploy.graphs.perception_graph import PerceptionGraph


class FlowPolicyGraph(nn.Module):
    """``HaloVLM.forward`` + ``HaloVLM.sample_actions`` with the noise as an **input**.

    Making the initial noise an input (instead of ``torch.randn`` inside the graph) keeps the
    graph deterministic, so exported runtimes can be checked against PyTorch bit-for-bit and
    apps control their own RNG. The ``num_steps`` Euler steps are unrolled at export time.

    Args:
        vlm: the built ``HaloVLM``.
        num_action_tokens: ``<halo_action>`` tokens in the prompt.
        num_steps: Euler steps (default: ``config.flow_num_ode_steps``).
    """

    def __init__(self, vlm: nn.Module, num_action_tokens: int = 1, num_steps: int | None = None) -> None:
        super().__init__()
        self.perception = PerceptionGraph(vlm, num_action_tokens)
        self.flow_decoder = vlm.flow_decoder
        self.chunk, self.action_dim = vlm.action_chunk_size, vlm.action_dim
        self.num_steps = int(num_steps or vlm.flow_num_ode_steps)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                states: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample action chunks with the unrolled Euler integrator.

        Args:
            images: frames ``[B, N, 3, H, W]``.
            input_ids: prompt ``[B, L]``.
            attention_mask: prompt mask ``[B, L]``.
            states: proprioception ``[B, S, state_dim]``.
            noise: initial noise ``[B, K, chunk * action_dim]``.

        Returns:
            ``[B, K, chunk, action_dim]``.
        """
        cond = self.perception(images, input_ids, attention_mask, states)          # [B, K, D]
        b, k, d = cond.shape
        cond = cond.reshape(b * k, d)
        x = noise.reshape(b * k, self.chunk * self.action_dim)
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            t = torch.full_like(x[:, 0], step * dt)
            x = x + self.flow_decoder(x, t, cond) * dt
        return x.reshape(b, k, self.chunk, self.action_dim)
