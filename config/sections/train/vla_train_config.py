"""VLATrainConfig: optimisation settings of a training run."""

from __future__ import annotations

from pydantic import Field

from config.sections.base import VLASection

from .checkpoint_config import CheckpointConfig
from .ray_section import RaySection


class VLATrainConfig(VLASection):
    """Optimisation settings.

    Attributes:
        epochs: training epochs.
        batch_size: samples per micro-step.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: global gradient-norm clip (0 disables).
        grad_accum_steps: micro-steps per optimiser step.
        action_loss_weight: weight of the flow-matching action loss.
        resume: checkpoint to resume from.
        checkpoint: checkpoint cadence.
        ray: Ray Train launcher settings.
    """

    ARGS = {"epochs": "epochs", "batch_size": "batch_size", "lr": "lr", "weight_decay": "weight_decay",
            "grad_clip": "grad_clip", "grad_accum_steps": "grad_accum_steps",
            "action_loss_weight": "action_loss_weight", "resume": "resume"}

    epochs: int = 5
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    action_loss_weight: float = 1.0
    resume: str | None = None              # checkpoint .pt to resume from
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    ray: RaySection = Field(default_factory=RaySection)
