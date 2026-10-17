"""BestKRouter: deterministic top-k gating for mixture-of-experts layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BestKRouter(nn.Module):
    """Route each token to its ``best_k`` highest-scoring experts (no exploration noise).

    The deterministic counterpart of :class:`NoiseBestKRouter`: logits outside the top-k are set
    to ``-inf`` before the softmax, so every token's weights are spread over exactly ``best_k``
    experts.

    Args:
        emb_dim: token width.
        num_exprts: number of routed experts.
        best_k: experts selected per token.
    """

    def __init__(self, emb_dim: int, num_exprts: int, best_k: int) -> None:
        super().__init__()
        self.best_k = best_k
        self.expert_proj = nn.Linear(emb_dim, num_exprts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weights [..., num_exprts], indices [..., best_k])`` for tokens ``x``."""
        logits = self.expert_proj(x)
        best_k_logits, idxs = torch.topk(logits, k=self.best_k, dim=-1)
        masked = torch.full_like(logits, float("-inf")).scatter(-1, idxs, best_k_logits)
        return F.softmax(masked, dim=-1), idxs
