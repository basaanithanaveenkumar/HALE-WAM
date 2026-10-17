"""WorldModelConfig: the DiT world model that predicts future frames (docs/world_model.md)."""

from __future__ import annotations

from pydantic import Field

from config.sections.base import VLASection

from .dit_config import DiTConfig


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
