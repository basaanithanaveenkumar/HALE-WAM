"""DecoderTransformer: the causal transformer stack of Halo-VLA (optionally with MoE feed-forwards)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .transformer_block import TransformerBlock


class DecoderTransformer(nn.Module):
    """A stack of causal :class:`TransformerBlock` layers followed by a final LayerNorm.

    Args:
        num_layers: number of blocks.
        emb_dim: model width.
        num_heads: attention heads per block.
        mlp_dim: hidden width of the dense feed-forward (used when ``use_moe`` is False).
        drop_fact: dropout probability.
        use_moe: use a DeepSeek-style mixture of experts as the feed-forward.
        moe_hid_scale: expert hidden width as a multiple of ``emb_dim``.
        moe_num_routed_experts: routed experts per MoE layer.
        moe_top_k: experts selected per token.
        moe_num_shared_experts: always-on shared experts.
        gradient_checkpointing: recompute activations in backward to save memory.
    """

    def __init__(
        self,
        num_layers: int = 16,
        emb_dim: int = 1024,
        num_heads: int = 32,
        mlp_dim: int = 512,
        drop_fact: float = 0.0,
        use_moe: bool = True,
        moe_hid_scale: float = 1.2,
        moe_num_routed_experts: int = 16,
        moe_top_k: int = 4,
        moe_num_shared_experts: int = 2,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    drop_fact=drop_fact,
                    causal_mask=True,
                    use_moe=use_moe,
                    moe_hid_scale=moe_hid_scale,
                    moe_num_routed_experts=moe_num_routed_experts,
                    moe_top_k=moe_top_k,
                    moe_num_shared_experts=moe_num_shared_experts,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, inputs: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run the stack.

        Args:
            inputs: embeddings ``[B, L, emb_dim]``.
            pad_mask: ``[B, L]`` with 1 for real tokens and 0 for padding.

        Returns:
            Normalised hidden states ``[B, L, emb_dim]``.
        """
        x = inputs
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Recompute activations during backward instead of storing them.
                # Saves ~activation memory per layer at the cost of one extra forward pass.
                x = checkpoint(layer, x, pad_mask, use_reentrant=False)
            else:
                x = layer(x, pad_mask=pad_mask)
        return self.norm(x)
