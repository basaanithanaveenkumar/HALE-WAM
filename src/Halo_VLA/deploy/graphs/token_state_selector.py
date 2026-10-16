"""Gather decoder states at placeholder tokens with static shapes (one-hot matmul)."""

from __future__ import annotations

import torch
from torch import nn


class TokenStateSelector(nn.Module):
    """States at the first ``count`` occurrences of ``token_id`` -> ``[B, count, D]``.

    Equivalent to ``HaloVLM.forward``'s ``nonzero``-based extraction (missing occurrences are
    zero rows, like its zero padding), but as ``one_hot [B, count, L] @ hidden [B, L, D]`` so
    the graph has no data-dependent shapes and lowers to TensorRT, MLX and NPU compilers.

    Args:
        token_id: placeholder id (e.g. ``<halo_action>``).
        count: number of occurrences to select (static).
    """

    def __init__(self, token_id: int, count: int) -> None:
        super().__init__()
        if count < 1:
            raise ValueError("count must be >= 1")
        self.token_id = int(token_id)
        self.count = int(count)

    def forward(self, text_hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        hit = input_ids == self.token_id                                   # [B, L]
        rank = torch.cumsum(hit.to(torch.int64), dim=1)                    # k-th hit has rank k
        slots = torch.arange(1, self.count + 1, device=input_ids.device).view(1, self.count, 1)
        one_hot = ((rank.unsqueeze(1) == slots) & hit.unsqueeze(1)).to(text_hidden.dtype)  # [B, K, L]
        return torch.matmul(one_hot, text_hidden)
