"""Latent extraction: what System 2 passes to System 1 (the key design choice of a dual system).

Every extractor maps System 2's encoding (see :meth:`System2.forward`) to ``tokens (B, K, D)``
and ``mask (B, K)`` (1 = real token). ``D`` is ``emb_dim`` except for ``middle_layer`` (the
decoder's raw width).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig


def _text_slice(enc: dict) -> slice:
    return slice(enc["num_prefix"], enc["num_prefix"] + enc["text_len"])


def _act_slice(enc: dict) -> slice:
    start = enc["num_prefix"] + enc["text_len"]
    return slice(start, start + enc["num_extra"])


class LatentExtractor(nn.Module):
    """Base class; subclasses implement :meth:`forward` for one latent type."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, enc: dict) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError


class ActTokenLatent(LatentExtractor):
    """Final states of the appended ``<ACT>`` queries (LCB, OpenHelix: 1 query; RoboDual: K)."""

    def __init__(self, width: int, with_language: bool = False) -> None:
        super().__init__(width)
        self.with_language = with_language

    def forward(self, enc):
        tokens = enc["hidden"][:, _act_slice(enc)]
        mask = enc["mask"][:, _act_slice(enc)]
        if self.with_language:  # RoboDual: task latent (instruction states) + action latent
            text = enc["hidden"][:, _text_slice(enc)]
            text_mask = enc["mask"][:, _text_slice(enc)]
            tokens, mask = torch.cat([tokens, text], 1), torch.cat([mask, text_mask], 1)
        return tokens, mask


class LastHiddenLatent(LatentExtractor):
    """Every last-layer state, vision and language (DP-VLA)."""

    def forward(self, enc):
        return enc["hidden"], enc["mask"]


class LanguageLatent(LatentExtractor):
    """Last-layer states of the instruction tokens only (DexVLA's language latent)."""

    def forward(self, enc):
        return enc["hidden"][:, _text_slice(enc)], enc["mask"][:, _text_slice(enc)]


class MiddleLayerLatent(LatentExtractor):
    """States of a middle decoder layer (GR00T-N1 style: more visual detail, cheaper)."""

    def forward(self, enc):
        if enc["layer_hidden"] is None:
            raise RuntimeError("middle_layer latent needs System 2 to capture a layer")
        return enc["layer_hidden"], enc["mask"]


class PooledLatent(LatentExtractor):
    """Multi-head attention pooling (MAP) of the last-layer states into one token (HiRT)."""

    def __init__(self, width: int, heads: int = 8) -> None:
        super().__init__(width)
        heads = heads if width % heads == 0 else 1
        self.probe = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, enc):
        hidden, mask = enc["hidden"], enc["mask"]
        probe = self.probe.expand(hidden.size(0), -1, -1)
        pooled, _ = self.attn(probe, hidden, hidden, key_padding_mask=mask == 0)
        return self.norm(pooled), torch.ones(hidden.size(0), 1, device=hidden.device, dtype=mask.dtype)


def build_latent_extractor(config: DualSystemConfig, emb_dim: int, layer_width: int) -> LatentExtractor:
    """The extractor for ``config.latent``."""
    kind = config.latent
    if kind in ("act_token", "multi_act"):
        return ActTokenLatent(emb_dim, with_language=kind == "multi_act" and config.multi_act_with_language)
    if kind == "last_hidden":
        return LastHiddenLatent(emb_dim)
    if kind == "language":
        return LanguageLatent(emb_dim)
    if kind == "pooled":
        return PooledLatent(emb_dim)
    return MiddleLayerLatent(layer_width)


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
            raise ValueError("alignment needs the latent and the instruction embeddings in the same space")
        return 1.0 - nn.functional.cosine_similarity(self.pool(tokens, mask), target.to(tokens.dtype), dim=-1).mean()

    def auxiliary_loss(self, tokens, mask, target_flat, target_mask_flat) -> torch.Tensor:
        pred = self.aux_head(self.pool(tokens, mask))
        err = (pred - target_flat).abs() * target_mask_flat
        return err.sum() / target_mask_flat.sum().clamp_min(1.0)
