"""
Model configuration for Halo-VLA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from exceptions import HaloVLAConfigError

if TYPE_CHECKING:
    from hale_core.config.sections.llm_backbone import LLMBackboneConfig


@dataclass
class HaloVLMConfig:
    """All hyper-parameters for HaloVLM in one place."""

    # Tokenizer / vocabulary (auto-set from special_tokens.json)
    # Using HuggingFaceTB/cosmo2-tokenizer (SmolLM tokenizer, 49152 base vocab)
    # which uses the same ChatML format as Qwen2.5 but is 3× smaller.
    vocab_size: int = 49156  # cosmo2-tokenizer (49152) + 4 custom tokens

    # Special token IDs (from config/special_tokens.json)
    image_token_id: int = 49152
    action_token_id: int = 49153
    state_token_id: int = 49154
    world_video_token_id: int = 49155

    # Shared embedding dimension
    emb_dim: int = 512

    # Factored-embedding bottleneck (ALBERT-style).
    # token_emb: vocab × emb_factor_dim   (shared with lm_head)
    # emb_proj:  emb_factor_dim → emb_dim (input side)
    # lm_proj:   emb_dim → emb_factor_dim (output side)
    # With emb_factor_dim=128: ~19.5M params vs ~155M for a full table.
    emb_factor_dim: int = 128

    # Vision encoder (ViT)
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    vit_num_layers: int = 6
    vit_num_heads: int = 16
    vit_mlp_dim: int = 512
    vit_drop: float = 0.0

    # Vision encoder selection (mirrors Hale-VLM's vision.encoder):
    #   "vit"    the built-in VisTransformer above (trained from scratch; default)
    #   "siglip" / "clip"  pretrained Hub towers (vision_model_id; img_size must match)
    #   "local"  any hale_core.nn.vision backbone by name (vision_backbone)
    vision_encoder: str = "vit"
    vision_model_id: str | None = None  # siglip/clip repo (default: base-patch16-224)
    vision_backbone: str = "dino_vit_small"  # for vision_encoder="local"
    vision_freeze: bool | None = None  # None: frozen for pretrained encoders, trained for "vit"
    vision_freeze_up_to_stage: int = -1  # "local": freeze stages 0..N, fine-tune the rest
    vision_pretrained_path: str | None = None  # "local": checkpoint for the backbone
    vision_num_tokens: int | None = None  # set by the encoder when the model is built

    # Decoder transformer
    dec_num_layers: int = 8
    dec_num_heads: int = 16
    dec_mlp_dim: int = 512
    dec_drop: float = 0.0

    # MoE (DeepseekMoE) — used inside each TransformerBlock
    use_moe: bool = True  # False → standard MLP FFN
    moe_hid_scale: float = 1.2  # hidden dim = round(emb_dim * scale)
    moe_num_routed_experts: int = 6
    moe_top_k: int = 2
    moe_num_shared_experts: int = 2

    # Positional embeddings
    max_position_embeddings: int = 2000

    # Language backbone. None keeps the built-in decoder above (token_emb + DecoderTransformer +
    # lm_head, cosmo2 tokenizer). Any name from hale_core.nn.llm.list_llm_backbones() - e.g.
    # "smollm2-360m", "qwen3-1.7b", "smollm3-3b", or the text tower of a natively multimodal model
    # ("qwen3.5-4b-base", "gemma4-e4b", "ministral3-3b", "cosmos-reason2-2b", "lfm2.5-vl-1.6b"),
    # or "hale-autoregressive" with llm_checkpoint - replaces it with that pretrained causal LM,
    # fine-tuned with LoRA / full / frozen. Its tokenizer then defines every token id.
    llm_backbone: str | None = None
    llm_model_id: str | None = None  # override the registered Hub repo / local path
    llm_checkpoint: str | None = None  # HaloDiffusionLLM .pt for hale-* backbones
    llm_finetune: str = "lora"  # lora | full | frozen
    llm_lora_r: int = 16
    llm_lora_alpha: int = 32
    llm_lora_dropout: float = 0.05
    llm_lora_target_modules: list[str] | None = None
    # The VLA's special tokens (<image>, <halo_action>, <state>, <halo_world_video>) are new to
    # a pretrained vocabulary; their embeddings must be learned, so train embeddings by default.
    llm_lora_train_embeddings: bool = True
    llm_dtype: str = "bfloat16"
    llm_gradient_checkpointing: bool = False
    llm_attn_implementation: str | None = "sdpa"  # Gemma 2/3/4: "eager" (logit soft-capping)

    # Dual-system VLA (src/Halo_VLA/dual_system, docs/dual_system.md): "none" keeps the
    # single-system Halo-VLA. A preset - basic, lcb, dp_vla, hirt, robodual, dexvla, openhelix -
    # or "custom" turns this model into System 2 and adds a fast System 1 policy fed by a latent.
    dual_system: str = "none"
    dual_system_options: dict | None = (
        None  # overrides, e.g. {"latent": "pooled", "async_steps": 4}
    )

    # Image projector (vision encoder -> decoder token space)
    proj_vision_dim: int | None = None  # defaults to emb_dim
    proj_llm_dim: int | None = None  # defaults to emb_dim
    # "mlp"                   3-layer MLP, one decoder token per ViT patch (default)
    # "qformer"               BLIP-2 style learned queries -> qformer_num_queries tokens
    # "gated_cross_attention" Flamingo style tanh-gated cross-attention -> gated_xattn_num_latents tokens
    proj_type: str = "mlp"

    # Q-Former (proj_type="qformer")
    qformer_num_queries: int = 32
    qformer_hidden_dim: int | None = None  # None -> min(proj_llm_dim, 768); divisible by num_heads
    qformer_num_layers: int = 2
    qformer_num_heads: int = 8
    qformer_cross_attention_freq: int = 1  # cross-attend to the image in every N-th layer
    qformer_ffn_mult: int = 4
    qformer_dropout: float = 0.0

    # Gated cross-attention (proj_type="gated_cross_attention")
    gated_xattn_num_latents: int = 32
    gated_xattn_hidden_dim: int | None = None
    gated_xattn_num_layers: int = 2
    gated_xattn_num_heads: int = 8
    gated_xattn_ffn_mult: int = 4
    gated_xattn_dropout: float = 0.0
    gated_xattn_gate_init: float = (
        1.0  # 0.0 = Flamingo identity start; >0 lets images flow at step 0
    )

    # Action decoder
    action_dim: int = 7  # output dims (e.g. 6-DOF + gripper)
    action_hidden_dims: tuple[int, ...] = (512, 256)
    action_chunk_size: int = 16  # predict N future steps at once
    action_dropout: float = 0.1
    action_use_layernorm: bool = True

    # State encoder
    state_dim: int = 32  # raw proprioceptive state dims
    state_hidden_dims: tuple[int, ...] = (256, 512)
    state_dropout: float = 0.1
    state_use_layernorm: bool = True

    # Flow matching action decoder
    flow_hidden_dim: int = 1024  # hidden dim in flow velocity network
    flow_time_embed_dim: int = 128  # dim of sinusoidal time embedding
    flow_num_ode_steps: int = 24  # Euler integration steps at inference

    # DiT visual heads (frame / depth / flow) — used by models.dit_frame_prediction
    dit_in_channels: int = 3  # default RGB; overridden per head when needed
    dit_out_channels: int = 3  # velocity has same layout as x (flow matching)

    # Visual prediction channels
    rgb_channels: int = 3
    depth_channels: int = 1
    flow_channels: int = 2

    dit_time_freq_dim: int = 512  # sinusoidal timestep embedding size — matches dit_hidden_size
    dit_patch_size: int = 8  # 224/8 = 28×28 = 784 tokens (4× finer than patch=16)
    dit_hidden_size: int = 512  # matches emb_dim — no context compression loss
    dit_depth: int = 8  # twice the velocity-field capacity
    dit_num_heads: int = 8  # hidden/heads = 64 (same ratio as before)
    dit_mlp_ratio: float = 4.0
    dit_max_resolution: int = 224  # pixel resolution of the DiT input (same as img_size)
    dit_rgb_in_channels: int = 3  # convenience mirrors for make_dit_config
    dit_depth_in_channels: int = 1
    dit_flow_in_channels: int = 2
    dit_flow_smooth_weight: float = 0.1
    dit_depth_recon_weight: float = 1.0
    dit_flow_photo_weight: float = 0.5
    dit_num_sample_steps: int = 75  # more Euler steps at inference → better quality

    # Gradient checkpointing — recomputes activations on backward instead of storing them.
    # Saves ~40% activation memory on the decoder at ~33% extra compute cost.
    gradient_checkpointing: bool = True

    # HaloVLM integration: DiT RGB / depth / flow heads (``VisualDiTPredictor``)
    enable_visual_dit: bool = True  # False → omit heads (e.g. old checkpoints)
    visual_loss_weight: float = 8.0  # increased to push more training signal into DiT head
    # Number of future frames the DiT head is asked to predict (configurable).
    # The dataloader samples num_sample_frames (context) + num_visual_predict_frames
    # (targets) and returns them as "images" and "future_frames" respectively.
    num_visual_predict_frames: int = 5  # focus capacity on near-future frames

    # System prompt
    system_prompt: str = (
        "You are a robotic VLA assistant. Given images and states, "
        "describe observations or output <halo_action> with a predicted trajectory."
    )

    def __post_init__(self) -> None:
        valid = ("mlp", "qformer", "gated_cross_attention")
        if self.proj_type not in valid:
            raise HaloVLAConfigError(f"proj_type must be one of {valid}, got {self.proj_type!r}")
        if self.vision_encoder not in ("vit", "siglip", "clip", "local"):
            raise HaloVLAConfigError(
                f"vision_encoder must be vit, siglip, clip or local, got {self.vision_encoder!r}"
            )
        if self.llm_finetune not in ("lora", "full", "frozen"):
            raise HaloVLAConfigError(
                f"llm_finetune must be lora, full or frozen, got {self.llm_finetune!r}"
            )

    @property
    def resolved_vision_freeze(self) -> bool:
        """Freeze pretrained towers by default; train the from-scratch ViT by default."""
        if self.vision_freeze is not None:
            return self.vision_freeze
        return self.vision_encoder != "vit"

    @property
    def uses_pretrained_llm(self) -> bool:
        """True when a registered LLM replaces the built-in decoder."""
        return self.llm_backbone is not None

    def llm_backbone_config(self) -> LLMBackboneConfig:
        """HaleBlocks ``LLMBackboneConfig`` for the ``llm_*`` fields (imports HaleBlocks lazily)."""
        try:
            from hale_core.config.sections.llm_backbone import LLMBackboneConfig, LoRASettings
        except ImportError as exc:  # pragma: no cover - depends on the environment
            from hale_core.exceptions import MissingDependencyError

            raise MissingDependencyError(
                "hale_core.config.sections.llm_backbone",
                "llm_backbone needs a HaleBlocks with the LLM registry: pip install -e '.[finetune]'",
            ) from exc
        return LLMBackboneConfig(
            name=self.llm_backbone,
            model_id=self.llm_model_id,
            checkpoint=self.llm_checkpoint,
            finetune=self.llm_finetune,
            lora=LoRASettings(
                r=self.llm_lora_r,
                alpha=self.llm_lora_alpha,
                dropout=self.llm_lora_dropout,
                target_modules=self.llm_lora_target_modules,
                train_embeddings=self.llm_lora_train_embeddings,
            ),
            dtype=self.llm_dtype,
            attn_implementation=self.llm_attn_implementation,
            gradient_checkpointing=self.llm_gradient_checkpointing,
        )

    @property
    def num_image_tokens_per_image(self) -> int:
        """Decoder tokens one image occupies: ViT patches for ``mlp``, else the connector size."""
        if self.proj_type == "qformer":
            return self.qformer_num_queries
        if self.proj_type == "gated_cross_attention":
            return self.gated_xattn_num_latents
        if self.vision_num_tokens is not None:  # measured from the configured encoder
            return self.vision_num_tokens
        return (self.img_size // self.patch_size) ** 2
