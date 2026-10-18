"""LatentBridge: projects the latent into System 1's space; hosts the two integration losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig
from exceptions import HaloVLAConfigError

from .bridge_projector_registry import BRIDGE_PROJECTORS
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

    Args:
        in_dim: latent width.
        out_dim: System 1's width.
        config: dual-system settings (``projector``, ``aux_weight``).
        action_flat: flattened action-chunk size predicted by the auxiliary head.
    """

    def __init__(
        self, in_dim: int, out_dim: int, config: DualSystemConfig, action_flat: int
    ) -> None:
        super().__init__()
        self.projector: nn.Module = BRIDGE_PROJECTORS.get(config.projector)(in_dim, out_dim)
        self.aux_head = (
            nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, action_flat))
            if config.aux_weight > 0
            else None
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project latent tokens ``(B, K, in_dim) -> (B, K, out_dim)``."""
        return self.projector(tokens)

    @staticmethod
    def pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean over the latent tokens: ``(B, K, D) -> (B, D)``."""
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def alignment_loss(
        self, tokens: torch.Tensor, mask: torch.Tensor, enc: dict[str, Any]
    ) -> torch.Tensor:
        """LCB-style cosine distance between the pooled latent and the instruction embeddings.

        Args:
            tokens: latent tokens ``(B, K, D)``.
            mask: latent mask ``(B, K)``.
            enc: System 2's encoding (needs ``reference_embeds``, ``mask``, ``num_prefix``, ``text_len``).

        Returns:
            A scalar loss.

        Raises:
            HaloVLAConfigError: if the latent and instruction embeddings differ in width.
        """
        text = enc["reference_embeds"][:, _text_slice(enc)]
        text_mask = enc["mask"][:, _text_slice(enc)]
        target = self.pool(text, text_mask)
        if target.size(-1) != tokens.size(-1):
            raise HaloVLAConfigError(
                "alignment needs the latent and the instruction embeddings in the same space"
            )
        return (
            1.0
            - nn.functional.cosine_similarity(
                self.pool(tokens, mask), target.to(tokens.dtype), dim=-1
            ).mean()
        )

    def auxiliary_loss(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        target_flat: torch.Tensor,
        target_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        """OpenHelix AUX: masked L1 between the action chunk predicted from the latent and the target.

        Args:
            tokens: latent tokens ``(B, K, D)``.
            mask: latent mask ``(B, K)``.
            target_flat: flattened target chunk ``(B, chunk * action_dim)``.
            target_mask_flat: validity mask of the same shape.

        Returns:
            A scalar loss.
        """
        pred = self.aux_head(self.pool(tokens, mask))
        err = (pred - target_flat).abs() * target_mask_flat
        return err.sum() / target_mask_flat.sum().clamp_min(1.0)
