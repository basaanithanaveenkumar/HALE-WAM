"""TransformerBlock: pre-norm attention + feed-forward (dense or mixture-of-experts) block."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.moe import DeepseekMoE

from .multi_head_attn import MultiHeadAttn


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))`` then ``x + ffn(norm(x))``.

    Args:
        emb_dim: model width.
        num_heads: attention heads.
        mlp_dim: hidden width of the dense feed-forward (``use_moe`` False).
        drop_fact: dropout probability.
        causal_mask: causal attention (decoder) or bidirectional (encoder).
        use_moe: use a DeepSeek-style mixture of experts as the feed-forward.
        moe_hid_scale: expert hidden width as a multiple of ``emb_dim``.
        moe_num_routed_experts: routed experts.
        moe_top_k: experts selected per token.
        moe_num_shared_experts: always-on shared experts.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        drop_fact: float = 0.0,
        causal_mask: bool = False,
        use_moe: bool = True,
        moe_hid_scale: float = 1.2,
        moe_num_routed_experts: int = 16,
        moe_top_k: int = 4,
        moe_num_shared_experts: int = 2,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttn(
            emb_dim=emb_dim, num_heads=num_heads, drop_fact=drop_fact, causal_mask=causal_mask
        )
        self.norm1 = nn.LayerNorm(emb_dim)

        self.use_moe = use_moe
        if use_moe:
            moe_hid_dim = round(emb_dim * moe_hid_scale)
            self.ffn = DeepseekMoE(
                emb_dim,
                moe_hid_dim,
                num_router_exprts=moe_num_routed_experts,
                best_k=moe_top_k,
                num_shared_exprts=moe_num_shared_experts,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(emb_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, emb_dim),
                nn.Dropout(drop_fact),
            )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, inputs: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply attention then the feed-forward, each with a residual connection.

        Args:
            inputs: ``[B, L, emb_dim]``.
            pad_mask: ``[B, L]`` with 1 for real tokens and 0 for padding.

        Returns:
            ``[B, L, emb_dim]``.
        """
        attn_output = self.attn(self.norm1(inputs), pad_mask=pad_mask)
        x = inputs + attn_output
        ffn_output = self.ffn(self.norm2(x))
        return x + ffn_output
