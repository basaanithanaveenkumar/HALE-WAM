"""Whole dual-system policy in one call (System 2 latent + System 1 sampler), for reference / parity."""

from __future__ import annotations

import torch
from torch import nn

from Halo_VLA.deploy.graphs.system1_policy_graph import System1PolicyGraph
from Halo_VLA.deploy.graphs.system2_latent_graph import System2LatentGraph


class DualSystemPolicyGraph(nn.Module):
    """``DualSystemVLA.predict_actions`` (no cached latent, no image mask) with noise as input.

    On a robot System 2 and System 1 run at different rates (``DualSystemController``), so they
    are deployed as separate graphs; this fused graph is the single-call reference used to
    validate those parts and for simple synchronous deployments.

    Args:
        dual: the built ``DualSystemVLA``.
        steps: System 1 sampling steps.
    """

    def __init__(self, dual: nn.Module, steps: int | None = None) -> None:
        super().__init__()
        self.system2 = System2LatentGraph(dual)
        self.system1 = System1PolicyGraph(dual, steps)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        states: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict an action chunk from an observation and prompt.

        Args:
            images: frames ``[B, N, 3, H, W]``.
            input_ids: prompt ``[B, L]``.
            attention_mask: prompt mask ``[B, L]``.
            states: proprioception ``[B, S, state_dim]``.
            noise: initial System 1 noise; ``None`` for regression.

        Returns:
            ``[B, chunk, action_dim]``.
        """
        latent, latent_mask = self.system2(images, input_ids, attention_mask, states)
        image = images[:, -1]  # newest frame (latest_frame without an image mask)
        state = (
            states[:, -1]
            if states.shape[1] > 0
            else states.new_zeros(states.shape[0], states.shape[2])
        )
        return self.system1(latent, latent_mask, image, state, noise)
