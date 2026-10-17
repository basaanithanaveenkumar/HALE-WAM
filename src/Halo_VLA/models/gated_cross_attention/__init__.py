"""Gated cross-attention connector (Flamingo style, Alayrac et al., 2022).

A set of learned latent tokens reads the vision features through *tanh-gated* cross-attention
and a *tanh-gated* feed-forward block::

    x = x + tanh(alpha_attn) * CrossAttn(LN(x), vision)
    x = x + tanh(alpha_ffn)  * FFN(LN(x))

The gates ``alpha`` are learned scalars. With ``gate_init=0`` the block is the identity at the
start of training (the LLM initially sees only the latents) and the image signal is blended in as
the gates open, which keeps a pretrained LLM undisturbed. A positive ``gate_init`` lets image
information flow from step 0, which converges faster at small learning rates.

Input  ``(B, N, vision_dim)``  ->  output ``(B, num_latents, llm_dim)``."""

from .gated_cross_attention_block import GatedCrossAttentionBlock  # noqa: F401
from .gated_cross_attention_projector import GatedCrossAttentionProjector  # noqa: F401

__all__ = [
    "GatedCrossAttentionBlock",
    "GatedCrossAttentionProjector",
]
