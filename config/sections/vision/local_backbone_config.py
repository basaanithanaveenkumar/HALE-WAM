"""LocalBackboneConfig: a ``hale_core.nn.vision`` backbone (``encoder: local``)."""

from __future__ import annotations

from config.sections.base import VLASection


class LocalBackboneConfig(VLASection):
    """A ``hale_core.nn.vision`` backbone (``encoder: local``)."""

    ARGS = {
        "backbone_name": "vision_backbone",
        "freeze_up_to_stage": "vision_freeze_up_to_stage",
        "pretrained_path": "vision_pretrained_path",
    }

    backbone_name: str = "dino_vit_small"
    freeze_up_to_stage: int = -1  # freeze stages 0..N, fine-tune the rest
    pretrained_path: str | None = None
