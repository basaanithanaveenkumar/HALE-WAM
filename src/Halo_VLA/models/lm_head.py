"""LMHead: the linear vocabulary projection of the decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class LMHead(nn.Module):
    """Linear vocabulary projection. Input is expected to be already normalized
    (DecoderTransformer applies the final LayerNorm before returning).

    Args:
        hidden_size: decoder width.
        vocab_size: number of output logits.
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, final_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states ``[..., hidden_size]`` to logits ``[..., vocab_size]``."""
        return self.lm_head(final_states)
