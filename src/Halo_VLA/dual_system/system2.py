"""System 2: the slow vision-language "brain" (Halo-VLA's VLM trunk)."""

from __future__ import annotations

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig


class System2(nn.Module):
    """Wraps a ``HaloVLM`` as the deliberate, low-frequency half of a dual-system VLA.

    It reads the instruction and the (possibly stale) camera frames and returns the decoder's
    hidden states, from which a latent is extracted for System 1. It owns the learnable
    ``<ACT>`` queries (LCB / RoboDual / OpenHelix): embeddings appended after the instruction,
    so in a causal decoder they attend to the whole multimodal context and their final states
    summarise it for the policy.

    Training modes (``DualSystemConfig.system2_training``):

    * ``frozen`` - no System 2 gradients at all (DP-VLA); the queries are fixed too.
    * ``prompt_tuning`` - System 2 frozen, only the ``<ACT>`` queries train (OpenHelix): the
      model's general abilities cannot degrade.
    * ``finetune`` - System 2 trains as already configured on the ``HaloVLM``: fully for the
      built-in decoder, or with the LLM backbone's ``llm_finetune`` policy (e.g. LoRA).

    Args:
        model: the ``HaloVLM``.
        config: dual-system settings.
    """

    def __init__(self, model: nn.Module, config: DualSystemConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        emb_dim = model.config.emb_dim
        n = config.num_act_tokens if config.uses_act_queries else 0
        self.act_queries = nn.Parameter(torch.randn(n, emb_dim) * 0.02) if n else None
        # System 1 replaces Halo-VLA's own action (flow) and video (DiT) heads: they take no part.
        for head in ("flow_decoder", "visual_predictor"):
            module = getattr(model, head, None)
            if module is not None:
                module.requires_grad_(False)
        mode = config.system2_training
        if mode in ("frozen", "prompt_tuning"):
            self.model.requires_grad_(False)
        if self.act_queries is not None:
            self.act_queries.requires_grad_(mode != "frozen")

    @property
    def frozen(self) -> bool:
        return self.config.system2_training in ("frozen", "prompt_tuning")

    def train(self, mode: bool = True) -> System2:
        super().train(mode)
        if self.frozen:
            self.model.eval()  # a frozen brain keeps dropout off / norms fixed
        return self

    def view_images(self, images: torch.Tensor) -> torch.Tensor:
        """Frames System 2 sees during training: shifted ``system2_frame_delay`` steps back.

        Slot ``i`` gets frame ``max(i - delay, 0)``, i.e. System 2 reasons about an older
        observation than the one System 1 acts on - the latency it has at deployment.
        """
        delay = self.config.system2_frame_delay
        if delay == 0 or images.size(1) == 1:
            return images
        index = (torch.arange(images.size(1), device=images.device) - delay).clamp_min(0)
        return images[:, index]

    def forward(self, batch: dict) -> dict:
        """Encode ``batch`` (images, input_ids, attention_mask, states) with the ``<ACT>`` queries."""
        b = batch["input_ids"].size(0)
        extra = None if self.act_queries is None else self.act_queries.unsqueeze(0).expand(b, -1, -1)
        layer = None
        if self.config.latent == "middle_layer":
            depth = len(self.model.decoder_layers())
            layer = self.config.middle_layer if self.config.middle_layer is not None else depth // 2
        with torch.set_grad_enabled(torch.is_grad_enabled() and not (self.frozen and extra is None)):
            out = self.model.encode_sequence(
                self.view_images(batch["images"]), batch["input_ids"], batch["attention_mask"],
                batch["states"], extra_embeds=extra, capture_layer=layer,
            )
        out["input_ids"] = batch["input_ids"]
        # Instruction reference for the alignment loss, in the latent's space (emb_dim): the
        # decoder inputs, mapped through the same bridge as the hidden states when an LLM is used.
        ref = out["input_embeds"]
        language = getattr(self.model, "language", None)
        if language is not None and isinstance(language.from_llm, nn.Linear):
            ref = language.from_llm(ref.to(language.from_llm.weight.dtype))
        out["reference_embeds"] = ref.detach()
        return out
