"""One training step of Halo-VLA as a single ``nn.Module`` (model + all losses)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_language_loss(logits, labels, num_prepended):
    """
    Cross-entropy loss on the text portion of the sequence.

    Args:
        logits:        [B, total_len, vocab_size]
        labels:        [B, seq_len]       (−100 for masked positions)
        num_prepended: int — number of image-patch tokens prepended
    """
    # The logits for text tokens start after the prepended image patches.
    # Shift by 1 for next-token prediction: predict token t+1 from position t.
    text_logits = logits[:, num_prepended:-1, :]  # [B, seq_len-1, V]
    target = labels[:, 1:]                         # [B, seq_len-1]

    # Truncate to the shorter of the two (in case of length mismatch)
    min_len = min(text_logits.size(1), target.size(1))
    text_logits = text_logits[:, :min_len, :]
    target = target[:, :min_len]

    loss = F.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        target.reshape(-1),
        ignore_index=-100,
    )
    return loss


@dataclass
class StepOutput:
    """Losses of one micro-step (``total`` is what gets back-propagated)."""

    total: torch.Tensor
    lang: torch.Tensor
    act: torch.Tensor
    vis: torch.Tensor
    vis_sub: dict[str, torch.Tensor] = field(default_factory=dict)
    action_hiddens: torch.Tensor | None = None
    """Decoder states at ``<halo_action>`` positions (evaluation samples actions from them)."""


class HaloVLATrainingStep(nn.Module):
    """``forward(batch) -> StepOutput``: the model forward **and** the three losses.

    Why a module: ``HaloVLM.compute_flow_loss`` and ``compute_visual_prediction_loss`` use the
    action-decoder and DiT parameters. Called outside ``forward`` they are invisible to
    DistributedDataParallel, which then cannot tell used from unused parameters (hangs or
    "marked ready twice" errors as soon as a branch is skipped, e.g. a batch without future
    frames). Wrapping the whole step makes ``DDP(step)`` see every parameter use.

    The single-process path uses the same module, so local and distributed training compute
    exactly the same losses.

    Args:
        model: the ``HaloVLM``.
        action_loss_weight: weight of the flow-matching action loss.
        visual_loss_weight: weight of the visual (DiT) prediction loss.
        perceptual_weight, ssim_weight, temporal_weight: visual sub-loss weights.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        action_loss_weight: float = 1.0,
        visual_loss_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.1,
        temporal_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.model = model
        self.action_loss_weight = action_loss_weight
        self.visual_loss_weight = visual_loss_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.temporal_weight = temporal_weight

    def forward(self, batch: dict) -> StepOutput:
        """``batch``: the collated dict already on the right device (``images``, ``input_ids``,
        ``attention_mask``, ``labels``, ``actions``, ``action_mask``, ``states`` and optionally
        ``image_mask`` / ``future_frames``)."""
        model, cfg = self.model, self.model.config
        images, input_ids = batch["images"], batch["input_ids"]
        logits, action_hiddens, visual_context_emb, world_video_query_hiddens = model(
            images=images,
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            states=batch["states"],
            image_mask=batch.get("image_mask"),
        )
        num_images = (input_ids == cfg.image_token_id).sum(dim=1).max().item()
        num_prepended = num_images * cfg.num_image_tokens_per_image
        lang = compute_language_loss(logits, batch["labels"], num_prepended)
        act = model.compute_flow_loss(action_hiddens, batch["actions"], batch["action_mask"])
        vis, vis_sub = model.compute_visual_prediction_loss(
            images,
            batch.get("image_mask"),
            visual_context_emb,
            world_video_query_hiddens=world_video_query_hiddens,
            future_frames=batch.get("future_frames"),
            perceptual_weight=self.perceptual_weight,
            ssim_weight=self.ssim_weight,
            temporal_weight=self.temporal_weight,
        )
        total = lang + self.action_loss_weight * act + self.visual_loss_weight * vis
        return StepOutput(total=total, lang=lang, act=act, vis=vis, vis_sub=vis_sub,
                          action_hiddens=action_hiddens)
