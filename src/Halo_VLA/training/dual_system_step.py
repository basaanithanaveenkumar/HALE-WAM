"""Training step for the dual-system VLA, in the same ``StepOutput`` shape as HaloVLATrainingStep."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.training_step import StepOutput


class DualSystemTrainingStep(nn.Module):
    """``forward(batch) -> StepOutput`` for a :class:`dual_system.DualSystemVLA`.

    Field mapping (so the training loop, logging and Ray reports stay unchanged): ``act`` = System 1
    policy loss, ``lang`` = System 2 language loss (0 unless ``lm_weight > 0``), ``vis`` = 0 (the
    dual system has no video head), ``vis_sub`` = ``{"aux": ..., "align": ...}``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> StepOutput:
        """Compute the dual-system losses for ``batch``.

        Args:
            batch: collated batch (``images``, ``input_ids``, ``attention_mask``, ``labels``, ``states``,
                ``actions``, ``action_mask``).

        Returns:
            A :class:`StepOutput` with the policy loss as ``act``.
        """
        losses = self.model(batch)
        zero = losses.policy.new_zeros(())
        return StepOutput(
            total=losses.total,
            lang=losses.lm,
            act=losses.policy,
            vis=zero,
            vis_sub={"loss_aux": losses.aux.detach(), "loss_align": losses.align.detach()},
        )
