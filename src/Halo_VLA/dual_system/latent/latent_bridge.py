"""LatentBridge: projects the latent into System 1's space; hosts the two integration losses."""

from __future__ import annotations

import torch
import torch.nn as nn
from exceptions import HaloVLAConfigError

from dual_system.config import DualSystemConfig

from .common import _text_slice


class LatentBridge(nn.Module):
    """Projects the latent into System 1's space; hosts the two integration losses.

    * projector: ``linear`` (OpenHelix) or 2-layer ``mlp`` from the latent width to System 1's.
    * :meth:`alignment_loss` (LCB's CLIP loss, adapted): cosine distance between the pooled
      latent and the instruction's own (frozen, detached) input embeddings, so fine-tuning
      System 2 cannot drift the latent away from the instruction's meaning. LCB aligns with a
      CLIP text encoder; here the instruction embedding of System 2 itself plays that role, so
      no extra model is needed.
    * :meth:`auxiliary_loss` (OpenHelix AUX): an MLP on the pooled latent predicts the action
      chunk, forcing the latent to carry visual / action information rather than only
      restating the instruction.
    """

    def __init__(self, in_dim: int, out_dim: int, config: DualSystemConfig, action_flat: int) -> None:
        super().__init__()
        if config.projector == "linear":
            self.projector: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            self.projector = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.aux_head = (
            nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, action_flat))
            if config.aux_weight > 0 else None
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(tokens)

    @staticmethod
    def pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean over the latent tokens: ``(B, K, D) -> (B, D)``."""
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def alignment_loss(self, tokens, mask, enc: dict) -> torch.Tensor:
        text = enc["reference_embeds"][:, _text_slice(enc)]
        text_mask = enc["mask"][:, _text_slice(enc)]
        target = self.pool(text, text_mask)
        if target.size(-1) != tokens.size(-1):
            raise HaloVLAConfigError("alignment needs the latent and the instruction embeddings in the same space")
        return 1.0 - nn.functional.cosine_similarity(self.pool(tokens, mask), target.to(tokens.dtype), dim=-1).mean()

    def auxiliary_loss(self, tokens, mask, target_flat, target_mask_flat) -> torch.Tensor:
        pred = self.aux_head(self.pool(tokens, mask))
        err = (pred - target_flat).abs() * target_mask_flat
        return err.sum() / target_mask_flat.sum().clamp_min(1.0)
