"""Single-system perception: observation + instruction -> action conditioning vectors."""

from __future__ import annotations

import torch
from torch import nn

from Halo_VLA.deploy.graphs.token_state_selector import TokenStateSelector
from Halo_VLA.deploy.graphs.vlm_sequence_encoder import VLMSequenceEncoder


class PerceptionGraph(nn.Module):
    """``HaloVLM.forward``'s ``action_hiddens`` as a standalone graph.

    Runs at camera rate on the robot; its output is the ``cond`` input of every flow-matching
    step of the action head (:class:`FlowActionDecoder`).

    Args:
        vlm: the built ``HaloVLM``.
        num_action_tokens: ``<halo_action>`` tokens in the prompt (one action chunk each).
    """

    def __init__(self, vlm: nn.Module, num_action_tokens: int = 1) -> None:
        super().__init__()
        self.encoder = VLMSequenceEncoder(vlm)
        self.select = TokenStateSelector(vlm.config.action_token_id, num_action_tokens)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                states: torch.Tensor) -> torch.Tensor:
        """Encode the observation and select the action-token states.

        Args:
            images: frames ``[B, N, 3, H, W]``.
            input_ids: prompt ``[B, L]``.
            attention_mask: prompt mask ``[B, L]``.
            states: proprioception ``[B, S, state_dim]``.

        Returns:
            ``[B, num_action_tokens, emb_dim]``.
        """
        hidden, _mask, num_prefix, _ = self.encoder(images, input_ids, attention_mask, states)
        return self.select(hidden[:, num_prefix:], input_ids)
