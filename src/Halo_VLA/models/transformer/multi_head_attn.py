"""MultiHeadAttn: multi-head self-attention built from per-head projections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from exceptions import HaloVLAConfigError

from .head_attn import HeadAttn


class MultiHeadAttn(nn.Module):
    """Multi-head self-attention over ``num_heads`` :class:`HeadAttn` heads.

    The heads keep separate ``Linear`` layers (checkpoint-compatible) but are evaluated together
    with ``F.scaled_dot_product_attention``, which avoids materialising the ``[B, H, L, L]`` matrix.

    Args:
        emb_dim: model width (divisible by ``num_heads``).
        num_heads: number of heads.
        drop_fact: dropout probability.
        causal_mask: forbid attending to later positions.
        return_attn_weights: forwarded to the heads (unused by the fused path).

    Raises:
        HaloVLAConfigError: if ``emb_dim`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_heads: int = 8,
        drop_fact: float = 0.0,
        causal_mask: bool = False,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()
        self.em_dim = emb_dim
        self.num_heads = num_heads
        if emb_dim % num_heads != 0:
            raise HaloVLAConfigError(
                f"emb_dim {emb_dim} must be divisible by num_heads {num_heads}"
            )

        self.head_size = emb_dim // num_heads
        self.causal_mask = causal_mask
        self.heads = nn.ModuleList(
            [
                HeadAttn(
                    emb_dim,
                    head_size=self.head_size,
                    drop_fact=drop_fact,
                    causal_mask=causal_mask,
                    return_attn_weights=return_attn_weights,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_fact)

    def forward(self, inputs: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attend over ``inputs``.

        Args:
            inputs: ``[B, L, D]``.
            pad_mask: ``[B, L]`` with 1 for real tokens and 0 for padding.

        Returns:
            ``[B, L, D]``.
        """
        B, L, D = inputs.shape
        # Project all heads at once → [B, H, L, head_size].
        # Weights come from the existing per-head Linear layers (checkpoint-compatible).
        Q = torch.stack([h.query_proj(inputs) for h in self.heads], dim=1)
        K = torch.stack([h.key_proj(inputs) for h in self.heads], dim=1)
        V = torch.stack([h.val_proj(inputs) for h in self.heads], dim=1)

        # Boolean attend-mask (True = this position is allowed to be attended to).
        attn_mask = None
        if self.causal_mask:
            # Lower triangle: token i can attend to tokens j <= i.
            causal = torch.tril(torch.ones(L, L, device=inputs.device, dtype=torch.bool))
            attn_mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        if pad_mask is not None:
            # pad_mask [B, L]: 1=real, 0=pad.  Mask key positions that are padding.
            key_mask = pad_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_mask = (attn_mask & key_mask) if attn_mask is not None else key_mask

        # F.scaled_dot_product_attention uses Flash Attention when available,
        # avoiding the explicit [B, H, L, L] attention matrix entirely.
        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # [B, H, L, head_size]

        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        return self.dropout(self.proj(out))
