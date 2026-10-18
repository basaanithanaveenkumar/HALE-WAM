"""NoiseBestKRouter: noisy top-k gating for mixture-of-experts layers (Shazeer et al., 2017)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseBestKRouter(nn.Module):
    """Route each token to its ``best_k`` experts, adding learned Gaussian noise while training.

    The noise scale is ``softplus(noise_linear(x))``, which encourages load balancing by letting
    the router explore (Shazeer et al., 2017); at eval time routing is deterministic.

    Args:
        emb_dim: token width.
        num_exprts: number of routed experts.
        best_k: experts selected per token.
    """

    def __init__(self, emb_dim: int, num_exprts: int, best_k: int) -> None:
        super().__init__()
        self.best_k = best_k
        self.bestk_layer = nn.Linear(emb_dim, num_exprts)
        self.noise_linear = nn.Linear(emb_dim, num_exprts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate the tokens ``x``.

        Args:
            x: tokens ``[..., emb_dim]``.

        Returns:
            ``(weights [..., num_exprts], indices [..., best_k])``; weights are zero outside the top-k.
        """
        logits = self.bestk_layer(x)
        noise_logits = self.noise_linear(x)
        # add the gaussian noise to logits
        if self.training:
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        best_k_logits, idxs = noisy_logits.topk(self.best_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, idxs, best_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, idxs
