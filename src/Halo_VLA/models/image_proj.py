import torch.nn as nn

from models.gated_cross_attention import GatedCrossAttentionProjector
from models.qformer import QFormer


PROJECTOR_TYPES = ("mlp", "qformer", "gated_cross_attention")


class ImageProjector(nn.Module):
    def __init__(self, vision_dim=512, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(vision_dim, llm_dim // 4),
                nn.LayerNorm(llm_dim // 4),
                nn.GELU(),
                nn.Linear(llm_dim//4, llm_dim // 2),
                nn.LayerNorm(llm_dim // 2),
                nn.GELU(),
                nn.Linear(llm_dim//2,llm_dim )
            )

    def forward(self, img_features):
        return self.proj(img_features)


def build_image_projector(config) -> nn.Module:
    """Build the vision -> decoder projector selected by ``config.proj_type``.

    * ``"mlp"``: :class:`ImageProjector`, one token per ViT patch.
    * ``"qformer"``: :class:`QFormer`, ``config.qformer_num_queries`` tokens per image.
    * ``"gated_cross_attention"``: :class:`GatedCrossAttentionProjector`,
      ``config.gated_xattn_num_latents`` tokens per image.
    """
    vision_dim = config.proj_vision_dim or config.emb_dim
    llm_dim = config.proj_llm_dim or config.emb_dim
    if config.proj_type == "mlp":
        return ImageProjector(vision_dim=vision_dim, llm_dim=llm_dim)
    if config.proj_type == "qformer":
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
    if config.proj_type == "gated_cross_attention":
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
    raise ValueError(f"Unknown proj_type {config.proj_type!r}; expected one of {PROJECTOR_TYPES}")

