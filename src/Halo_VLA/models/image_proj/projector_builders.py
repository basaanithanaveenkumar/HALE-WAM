"""Builders that register each projector kind in :data:`IMAGE_PROJECTORS`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from models.gated_cross_attention import GatedCrossAttentionProjector
from models.qformer import QFormer

from .image_projector import ImageProjector
from .projector_registry import IMAGE_PROJECTORS

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig


def _widths(config: HaloVLMConfig) -> tuple[int, int]:
    """``(vision_dim, llm_dim)``: the projector's input and output widths."""
    return config.proj_vision_dim or config.emb_dim, config.proj_llm_dim or config.emb_dim


@IMAGE_PROJECTORS.register_module(name="mlp")
def build_mlp_projector(config: HaloVLMConfig) -> nn.Module:
    """:class:`ImageProjector`: one token per ViT patch."""
    vision_dim, llm_dim = _widths(config)
    return ImageProjector(vision_dim=vision_dim, llm_dim=llm_dim)


@IMAGE_PROJECTORS.register_module(name="qformer")
def build_qformer_projector(config: HaloVLMConfig) -> nn.Module:
    """:class:`QFormer`: ``config.qformer_num_queries`` tokens per image."""
    vision_dim, llm_dim = _widths(config)
    return QFormer(
        vision_dim,
        llm_dim,
        num_queries=config.qformer_num_queries,
        hidden_dim=config.qformer_hidden_dim,
        num_layers=config.qformer_num_layers,
        num_heads=config.qformer_num_heads,
        cross_attention_freq=config.qformer_cross_attention_freq,
        ffn_mult=config.qformer_ffn_mult,
        dropout=config.qformer_dropout,
    )


@IMAGE_PROJECTORS.register_module(name="gated_cross_attention")
def build_gated_cross_attention_projector(config: HaloVLMConfig) -> nn.Module:
    """:class:`GatedCrossAttentionProjector`: ``config.gated_xattn_num_latents`` tokens per image."""
    vision_dim, llm_dim = _widths(config)
    return GatedCrossAttentionProjector(
        vision_dim,
        llm_dim,
        num_latents=config.gated_xattn_num_latents,
        hidden_dim=config.gated_xattn_hidden_dim,
        num_layers=config.gated_xattn_num_layers,
        num_heads=config.gated_xattn_num_heads,
        ffn_mult=config.gated_xattn_ffn_mult,
        dropout=config.gated_xattn_dropout,
        gate_init=config.gated_xattn_gate_init,
    )
