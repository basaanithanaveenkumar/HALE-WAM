"""VLAModelConfig: halo-VLA architecture; nested sections as in Hale-VLM (``vision``, ``llm``)."""

from __future__ import annotations

from pydantic import Field

from config.sections.base import VLASection
from config.sections.dual_system import DualSystemSection
from config.sections.llm import LLMConfig
from config.sections.vision import VisionConfig

from .action_config import ActionConfig
from .decoder_config import DecoderConfig
from .moe_config import MoEConfig
from .world_model_config import WorldModelConfig


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
