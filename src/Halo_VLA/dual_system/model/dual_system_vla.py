"""DualSystemVLA: the newest real frame of every sample: ``(B, N, 3, H, W) -> (B, 3, H, W)``."""

from __future__ import annotations

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig
from dual_system.latent import LatentBridge, build_latent_extractor
from dual_system.system1 import System1Policy
from dual_system.system2 import System2

from .dual_system_losses import DualSystemLosses


def latest_frame(images: torch.Tensor, image_mask: torch.Tensor | None) -> torch.Tensor:
    """The newest real frame of every sample: ``(B, N, 3, H, W) -> (B, 3, H, W)``."""
    if image_mask is None:
        return images[:, -1]
    last = (image_mask.long().cumsum(1).argmax(1)).clamp_min(0)  # index of the last 1
    return images[torch.arange(images.size(0), device=images.device), last]


def latest_state(states: torch.Tensor, state_dim: int) -> torch.Tensor:
    """The newest proprioceptive state ``(B, state_dim)`` (zeros when the sample has none)."""
    if states.size(1) == 0:
        return states.new_zeros(states.size(0), state_dim)
    return states[:, -1]


def action_targets(
    actions: torch.Tensor, mask: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first ``chunk`` actions and their validity mask, zero-padded to ``chunk`` steps."""
    b, t, d = actions.shape
    if t < chunk:
        actions = torch.cat([actions, actions.new_zeros(b, chunk - t, d)], 1)
        mask = torch.cat([mask, mask.new_zeros(b, chunk - t)], 1)
    return actions[:, :chunk], mask[:, :chunk].to(actions.dtype)


class DualSystemVLA(nn.Module):
    """System 2 (``HaloVLM`` trunk) -> latent -> projector -> System 1 (fast policy).

    Training follows each design's recipe (``DualSystemConfig``): the policy loss of System 1,
    plus optional LCB alignment, OpenHelix auxiliary action and System 2 language-modelling
    losses. With ``prealign_steps > 0`` (OpenHelix) System 1 is frozen for the first steps while
    the projector and prompt align, then trained jointly - call :meth:`set_training_step` every
    optimiser step and :meth:`mask_frozen_gradients` before the optimiser step. Inference runs
    System 2 rarely and System 1 every step
    (:class:`dual_system.controller.DualSystemController`).

    Args:
        system2_model: the ``HaloVLM`` used as System 2.
        config: dual-system settings.
    """

    def __init__(self, system2_model: nn.Module, config: DualSystemConfig) -> None:
        super().__init__()
        self.config = config
        vcfg = system2_model.config
        self.state_dim = vcfg.state_dim
        self.chunk, self.action_dim = vcfg.action_chunk_size, vcfg.action_dim
        self.system2 = System2(system2_model, config)
        self.latent = build_latent_extractor(config, vcfg.emb_dim, system2_model.layer_width)
        self.bridge = LatentBridge(
            self.latent.width, config.system1_dim, config, self.chunk * self.action_dim
        )
        self.system1 = System1Policy(
            config, vcfg.img_size, vcfg.state_dim, self.chunk, vcfg.action_dim
        )
        if config.system1_checkpoint:
            state = torch.load(config.system1_checkpoint, map_location="cpu", weights_only=True)
            self.system1.load_state_dict(state.get("system1", state))
        self._stage = None
        self.set_training_step(0)

    # ------------------------------------------------------------------ staging
    @property
    def in_prealignment(self) -> bool:
        """Whether System 1 is still frozen (OpenHelix stage 1)."""
        return self._stage == "prealign"

    def set_training_step(self, step: int) -> None:
        """OpenHelix two-stage schedule: System 1 is held fixed for the first ``prealign_steps``."""
        self._stage = "prealign" if step < self.config.prealign_steps else "joint"

    def mask_frozen_gradients(self) -> None:
        """Call between ``backward()`` and ``optimizer.step()``: in the pre-alignment stage drop
        System 1's gradients so only the prompt / projector (and System 2 if trained) move.

        Freezing by gradient masking rather than ``requires_grad_`` keeps the parameter set fixed,
        which DistributedDataParallel requires (it registers parameters once, at wrap time).
        """
        if self.in_prealignment:
            for p in self.system1.parameters():
                p.grad = None

    def optimizer_parameters(self) -> list[nn.Parameter]:
        """Parameters that train in any stage (frozen System 2 weights excluded)."""
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ forward pieces
    def encode_latent(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """System 2 -> latent tokens projected to System 1's width (+ the raw encoding)."""
        enc = self.system2(batch)
        tokens, mask = self.latent(enc)
        enc["latent_tokens"], enc["latent_mask"] = tokens, mask
        return self.bridge(tokens.float()), mask, enc

    def system1_condition(
        self, latent: torch.Tensor, latent_mask: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Condition System 1 on a latent and the batch's newest frame and state.

        Args:
            latent: projected latent ``(B, K, system1_dim)``.
            latent_mask: its mask ``(B, K)``.
            batch: the batch (``images``, optional ``image_mask``, ``states``).

        Returns:
            ``(condition tokens, condition mask)``.
        """
        image = latest_frame(batch["images"], batch.get("image_mask"))
        state = latest_state(batch["states"], self.state_dim)
        return self.system1.condition(latent, latent_mask, image, state)

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict) -> DualSystemLosses:
        """All loss terms for one batch (``images, input_ids, attention_mask, labels, states,
        actions, action_mask`` and optional ``image_mask``)."""
        latent, latent_mask, enc = self.encode_latent(batch)
        cond, cond_mask = self.system1_condition(latent, latent_mask, batch)
        actions, mask = action_targets(
            batch["actions"].float(), batch["action_mask"].float(), self.chunk
        )
        policy = self.system1.loss(cond, cond_mask, actions, mask)
        zero = policy.new_zeros(())
        aux = align = lm = zero
        cfg = self.config
        if cfg.aux_weight > 0:
            flat_mask = mask.unsqueeze(-1).expand_as(actions).reshape(actions.size(0), -1)
            aux = self.bridge.auxiliary_loss(
                enc["latent_tokens"].float(),
                enc["latent_mask"],
                actions.reshape(actions.size(0), -1),
                flat_mask,
            )
        if cfg.align_weight > 0:
            align = self.bridge.alignment_loss(
                enc["latent_tokens"].float(), enc["latent_mask"], enc
            )
        if cfg.lm_weight > 0:
            lm = self._language_loss(enc, batch["labels"])
        total = policy + cfg.aux_weight * aux + cfg.align_weight * align + cfg.lm_weight * lm
        return DualSystemLosses(total=total, policy=policy, aux=aux, align=align, lm=lm)

    @staticmethod
    def _language_loss(enc: dict, labels: torch.Tensor) -> torch.Tensor:
        """Next-token loss of System 2 on the text (image prefix and <ACT> queries excluded)."""
        start = enc["num_prefix"]
        logits = enc["logits"][:, start : start + enc["text_len"]]
        return nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_actions(
        self, batch: dict, latent: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Action chunk ``(B, chunk, action_dim)``; pass a cached ``(latent, mask)`` to skip System 2."""
        if latent is None:
            latent_tokens, latent_mask, _ = self.encode_latent(batch)
        else:
            latent_tokens, latent_mask = latent
        cond, cond_mask = self.system1_condition(latent_tokens, latent_mask, batch)
        return self.system1.sample(cond, cond_mask)
