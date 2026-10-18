"""SelfAttn: single-head self-attention with full-width projections."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    """Single-head scaled dot-product self-attention (query / key / value are ``emb_dim`` wide).

    Args:
        emb_dim: model width.
        return_attn_weights: also return the ``[B, L, L]`` attention weights.
    """

    def __init__(self, emb_dim: int, return_attn_weights: bool = False) -> None:
        super().__init__()
        self.em_dim = emb_dim

        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.val_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)
        self.return_attn_weights = return_attn_weights

    def forward(
        self,
        inputs: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend over ``inputs``.

        Args:
            inputs: ``[B, L, D]``.
            causal_mask: ``[L, L]`` with 0 where attention is forbidden.
            pad_mask: ``[B, L]`` with 1 for real tokens and 0 for padding.

        Returns:
            ``[B, L, D]``, plus the attention weights when ``return_attn_weights`` is set.
        """
        Query = self.query_proj(inputs)  # [B, Seq_len, D]
        Key = self.key_proj(inputs)  # [B, Seq_len, D]
        Val = self.val_proj(inputs)  # [B, Seq_len, D]

        attn_scores = (
            torch.matmul(Query, Key.transpose(-2, -1)) * self.scale
        )  # [B, Seq_len, Seq_len]

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # pad_mask: [B, Seq_len] with 1=real, 0=pad; mask out pad key positions
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(pad_mask.unsqueeze(1) == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Seq_len, Seq_len]
        attn_output = torch.matmul(attn_weights, Val)  # [B, Seq_len, D]

        if self.return_attn_weights:
            return attn_output, attn_weights
        return attn_output
