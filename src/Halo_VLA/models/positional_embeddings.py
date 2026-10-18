"""Sinusoidal positional embeddings of the original Transformer (Vaswani et al., 2017)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sine / cosine positions added to a token sequence.

    Args:
        d_model: embedding width (even).
        max_len: longest supported sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even channels
        pe[:, 1::2] = torch.cos(position * div_term)  # odd channels
        # A buffer, not a parameter: positions are fixed.
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the positions to ``x``.

        Args:
            x: tokens ``[batch_size, seq_len, d_model]``.

        Returns:
            ``x`` plus the first ``seq_len`` positional rows.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].detach()
