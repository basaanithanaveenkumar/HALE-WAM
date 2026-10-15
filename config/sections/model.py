"""``model``: the Halo-VLA architecture (decoder, vision, LLM, action / world-model heads, dual system)."""

from __future__ import annotations

from pydantic import Field

from config.sections.base import VLASection
from config.sections.dual_system import DualSystemSection
from config.sections.llm import LLMConfig
from config.sections.vision import VisionConfig


class DecoderConfig(VLASection):
    """The built-in decoder transformer (unused when ``model.llm.backbone`` is set)."""

    OVERRIDES = {"num_layers": "dec_num_layers", "num_heads": "dec_num_heads", "mlp_dim": "dec_mlp_dim",
                 "dropout": "dec_drop"}

    num_layers: int = 8
    num_heads: int = 16
    mlp_dim: int = 512
    dropout: float = 0.0


class MoEConfig(VLASection):
    """DeepSeekMoE feed-forwards inside the built-in decoder."""

    OVERRIDES = {"enabled": "use_moe", "hid_scale": "moe_hid_scale", "num_routed_experts": "moe_num_routed_experts",
                 "top_k": "moe_top_k", "num_shared_experts": "moe_num_shared_experts"}

    enabled: bool = True
    hid_scale: float = 1.2
    num_routed_experts: int = 6
    top_k: int = 2
    num_shared_experts: int = 2


class FlowConfig(VLASection):
    """Flow-matching action decoder."""

    OVERRIDES = {"hidden_dim": "flow_hidden_dim", "time_embed_dim": "flow_time_embed_dim",
                 "num_ode_steps": "flow_num_ode_steps"}

    hidden_dim: int = 1024
    time_embed_dim: int = 128
    num_ode_steps: int = 24


class ActionConfig(VLASection):
    """Action / proprioception spaces and their encoders."""

    ARGS = {"action_dim": "action_dim", "state_dim": "state_dim", "chunk_size": "action_chunk_size"}
    OVERRIDES = {"hidden_dims": "action_hidden_dims", "dropout": "action_dropout",
                 "use_layernorm": "action_use_layernorm", "state_hidden_dims": "state_hidden_dims",
                 "state_dropout": "state_dropout"}

    action_dim: int = 32
    state_dim: int = 32
    chunk_size: int = 16
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    use_layernorm: bool = True
    state_hidden_dims: list[int] = Field(default_factory=lambda: [256, 512])
    state_dropout: float = 0.1
    flow: FlowConfig = Field(default_factory=FlowConfig)


class DiTConfig(VLASection):
    """DiT video / depth / flow predictor."""

    OVERRIDES = {"patch_size": "dit_patch_size", "hidden_size": "dit_hidden_size", "depth": "dit_depth",
                 "num_heads": "dit_num_heads", "mlp_ratio": "dit_mlp_ratio", "num_sample_steps": "dit_num_sample_steps"}

    patch_size: int = 8
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_sample_steps: int = 75


class WorldModelConfig(VLASection):
    """The DiT world model that predicts future frames (docs/world_model.md)."""

    ARGS = {"loss_weight": "visual_loss_weight", "num_predict_frames": "num_predict_frames",
            "perceptual_weight": "perceptual_weight", "ssim_weight": "ssim_weight",
            "temporal_weight": "temporal_weight"}
    OVERRIDES = {"enabled": "enable_visual_dit"}

    enabled: bool = True
    loss_weight: float | None = None       # None = HaloVLMConfig.visual_loss_weight (8.0)
    num_predict_frames: int = 5
    perceptual_weight: float = 0.1
    ssim_weight: float = 0.4
    temporal_weight: float = 0.2
    dit: DiTConfig = Field(default_factory=DiTConfig)


class VLAModelConfig(VLASection):
    """Halo-VLA architecture; nested sections as in Hale-VLM (``vision``, ``llm``)."""

    OVERRIDES = {"emb_dim": "emb_dim", "emb_factor_dim": "emb_factor_dim",
                 "max_position_embeddings": "max_position_embeddings",
                 "gradient_checkpointing": "gradient_checkpointing"}

    emb_dim: int = 512
    emb_factor_dim: int = 128
    max_position_embeddings: int = 2000
    gradient_checkpointing: bool = True
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    dual_system: DualSystemSection = Field(default_factory=DualSystemSection)
