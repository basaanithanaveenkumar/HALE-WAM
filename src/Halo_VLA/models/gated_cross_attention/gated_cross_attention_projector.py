"""GatedCrossAttentionProjector: learned latents + gated cross-attention blocks, projected to the LLM hidden size."""

from __future__ import annotations

import torch
import torch.nn as nn
from exceptions import HaloVLAConfigError

from .gated_cross_attention_block import GatedCrossAttentionBlock


class GatedCrossAttentionProjector(nn.Module):
    """Learned latents + gated cross-attention blocks, projected to the LLM hidden size.

    Args:
        vision_dim: channel size of the vision encoder tokens.
        llm_dim: hidden size of the language model.
        num_latents: number of output tokens.
        hidden_dim: internal width; ``None`` uses ``min(llm_dim, 768)``.
        num_layers: number of gated blocks.
        num_heads: attention heads (must divide ``hidden_dim``).
        ffn_mult: feed-forward expansion factor.
        dropout: dropout inside attention and the FFN.
        gate_init: initial value of every gate before ``tanh`` (0 = Flamingo identity start).
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        *,
        num_latents: int = 64,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or min(llm_dim, 768)
        if hidden_dim % num_heads:
            raise HaloVLAConfigError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)
        self.memory_norm = nn.LayerNorm(vision_dim)
        self.blocks = nn.ModuleList(
            GatedCrossAttentionBlock(
                hidden_dim, vision_dim, num_heads, ffn_mult, dropout, gate_init
            )
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, llm_dim)

    def output_tokens(self, num_input_tokens: int | None = None) -> int:
        """Number of tokens produced per image (independent of the input length)."""
        del num_input_tokens
        return self.num_latents

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(vision_features)
        x = self.latents.expand(vision_features.shape[0], -1, -1).to(memory.dtype)
        for block in self.blocks:
            x = block(x, memory)
        return self.out_proj(self.out_norm(x))
