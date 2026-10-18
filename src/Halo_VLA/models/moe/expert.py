"""Expert: a SwiGLU feed-forward network, the unit of a mixture-of-experts layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """SwiGLU feed-forward: ``w2(silu(w1 x) * w3 x)``.

    Args:
        emb_dim: input and output width.
        hid_dim: hidden width.
        dropout: dropout probability on the output.
    """

    def __init__(self, emb_dim: int, hid_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(emb_dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, emb_dim, bias=False)
        self.w3 = nn.Linear(emb_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expert to ``x`` of shape ``[..., emb_dim]``."""
        # SwiGLU (DeepSeekMoE's choice) gates the hidden units; it is deliberately not GELU.
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
