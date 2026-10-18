"""A pretrained, fine-tunable LLM as Halo-VLA's decoder (replaces the built-in DecoderTransformer)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from config.tokens import get_all_custom_tokens

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig


class PretrainedLanguageDecoder(nn.Module):
    """Any registered HaleBlocks LLM backbone, bridged to the VLA's ``emb_dim`` space.

    The VLA's other parts - ViT + image projector, state encoder, flow-matching action decoder,
    DiT video predictor - keep working in ``emb_dim``. This module:

    * embeds text with the LLM's own embedding table (``hidden_size`` wide);
    * maps image / state embeddings ``emb_dim -> hidden_size`` (:attr:`to_llm`) before they are
      spliced into the sequence;
    * runs the LLM causally (its own positions, attention, final norm, LM head);
    * maps the final hidden states ``hidden_size -> emb_dim`` (:attr:`from_llm`) for the action
      and video heads, which read them at ``<halo_action>`` / ``<halo_world_video>`` / image slots.

    The VLA's special tokens are added to the LLM's tokenizer **before** fine-tuning is applied
    (so LoRA / freezing covers the resized embeddings correctly), and their ids, the vocabulary
    size and the tokenizer are written back into ``config`` - the dataloaders must use
    :attr:`tokenizer` so token ids match.

    Args:
        config: ``HaloVLMConfig`` with ``llm_backbone`` set; updated in place with token ids.
    """

    def __init__(self, config: HaloVLMConfig) -> None:
        super().__init__()
        from hale_core.nn.llm import build_language_backbone  # lazy: optional HaleBlocks extra

        tokens = get_all_custom_tokens()  # <image>, <halo_action>, <state>, <halo_world_video>
        self.llm, ids = build_language_backbone(config.llm_backbone_config(), special_tokens=tokens)
        (
            config.image_token_id,
            config.action_token_id,
            config.state_token_id,
            config.world_video_token_id,
        ) = (ids[t] for t in tokens)
        config.vocab_size = self.llm.vocab_size
        hidden = self.llm.hidden_size
        # Trainable bridges (fp32); identity when the widths already agree.
        self.to_llm = (
            nn.Linear(config.emb_dim, hidden) if hidden != config.emb_dim else nn.Identity()
        )
        self.from_llm = (
            nn.Linear(hidden, config.emb_dim) if hidden != config.emb_dim else nn.Identity()
        )
        for layer in (self.to_llm, self.from_llm):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    @property
    def tokenizer(self) -> Any:
        """The LLM's tokenizer, including the VLA's special tokens."""
        return self.llm.tokenizer

    @property
    def hidden_size(self) -> int:
        """Width of the LLM's hidden states."""
        return self.llm.hidden_size

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` -> ``(B, L, hidden_size)`` LLM token embeddings."""
        return self.llm.embed_tokens(input_ids)

    def project_in(self, x: torch.Tensor) -> torch.Tensor:
        """VLA-space embeddings (images, states) ``(..., emb_dim)`` -> ``(..., hidden_size)``."""
        return self.to_llm(x)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the LLM over the spliced sequence.

        Args:
            inputs_embeds: ``(B, T, hidden_size)`` image + text/state embeddings.
            attention_mask: ``(B, T)`` 1 = real token.
            token_ids: ``(B, T)`` ids aligned with ``inputs_embeds`` (image token at image slots).
                Backbones whose layers read token identity (Gemma 4 per-layer embeddings) need
                them because the spliced embeddings cannot be mapped back to ids; others ignore them.

        Returns:
            ``logits (B, T, V)`` and hidden states mapped back to ``(B, T, emb_dim)``.
        """
        dtype = self.llm.embedding_parameters()[0].dtype  # the LLM's weight dtype (e.g. bfloat16)
        out = self.llm(
            inputs_embeds=inputs_embeds.to(dtype),
            attention_mask=attention_mask,
            token_ids=token_ids,
        )
        hidden = out.hidden_states
        if isinstance(self.from_llm, nn.Linear):
            hidden = hidden.to(self.from_llm.weight.dtype)
        return out.logits.float(), self.from_llm(hidden)
