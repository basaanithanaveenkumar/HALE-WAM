"""Q-Former: learned queries that compress vision tokens into a fixed set of LLM tokens.

BLIP-2 style (Li et al., 2023), vision-only: ``num_queries`` learned query vectors run through
transformer layers. Every layer lets the queries attend to each other (self-attention); every
``cross_attention_freq``-th layer also lets them attend to the frozen vision features
(cross-attention). The result is projected to the LLM hidden size.

Input  ``(B, N, vision_dim)``  ->  output ``(B, num_queries, llm_dim)``."""

from .qformer import QFormer  # noqa: F401
from .qformer_layer import QFormerLayer  # noqa: F401

__all__ = [
    "QFormer",
    "QFormerLayer",
]
