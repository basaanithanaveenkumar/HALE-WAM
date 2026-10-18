"""Traceable ``HaloVLM.encode_sequence``: embeddings -> decoder hidden states (no task heads)."""

from __future__ import annotations

import torch
from torch import nn

from Halo_VLA.deploy.graphs.static_prompt_embedder import StaticPromptEmbedder


class VLMSequenceEncoder(nn.Module):
    """Same computation as ``HaloVLM.encode_sequence`` on top of :class:`StaticPromptEmbedder`.

    Returns ``(hidden [B, T, emb_dim], mask [B, T], num_prefix, layer_hidden | None)`` where
    ``T = num_prefix + L (+ K extra)``. The language-model logits are not returned: deployment
    graphs feed the action heads, not text generation (the LLM is exported separately as a
    Hugging Face checkpoint).

    Args:
        vlm: the built ``HaloVLM``.
        capture_layer: decoder layer whose output is also returned (dual-system
            ``middle_layer`` latent); captured with a forward hook exactly like the original.
    """

    def __init__(self, vlm: nn.Module, capture_layer: int | None = None) -> None:
        super().__init__()
        self.embedder = StaticPromptEmbedder(vlm)
        self.language = vlm.language
        self.decoder = vlm.decoder_transformer
        self.pos_embed = vlm.pos_embed if vlm.language is None else None
        self.image_token_id = int(vlm.config.image_token_id)
        self._captured: dict[str, torch.Tensor] = {}
        self.capture_layer = capture_layer
        if capture_layer is not None:
            vlm.decoder_layers()[capture_layer].register_forward_hook(self._grab)

    def _grab(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
        self._captured["h"] = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        states: torch.Tensor,
        extra_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
        """Encode the observation and prompt with the VLM decoder.

        Args:
            images: frames ``[B, N, 3, H, W]``.
            input_ids: prompt ``[B, L]``.
            attention_mask: prompt mask ``[B, L]``.
            states: proprioception ``[B, S, state_dim]``.
            extra_embeds: optional ``[B, K, emb_dim]`` embeddings appended after the text.

        Returns:
            ``(hidden, mask, num_prefix, layer_hidden)`` as described in the class docstring.
        """
        b = input_ids.shape[0]
        combined, num_prefix = self.embedder(images, input_ids, states)
        mask = torch.cat(
            [
                torch.ones(b, num_prefix, dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask,
            ],
            dim=1,
        )
        num_extra = 0
        if extra_embeds is not None:
            extra = (
                self.language.project_in(extra_embeds)
                if self.language is not None
                else extra_embeds
            )
            combined = torch.cat([combined, extra.to(combined.dtype)], dim=1)
            mask = torch.cat(
                [mask, torch.ones(b, extra.shape[1], dtype=mask.dtype, device=mask.device)], dim=1
            )
            num_extra = extra.shape[1]
        self._captured.clear()
        if self.language is not None:
            token_ids = torch.cat(
                [input_ids.new_full((b, num_prefix), self.image_token_id), input_ids], dim=1
            )
            if num_extra:
                pad = getattr(self.language.tokenizer, "pad_token_id", None) or 0
                token_ids = torch.cat([token_ids, input_ids.new_full((b, num_extra), pad)], dim=1)
            _logits, hidden = self.language(combined, mask, token_ids)
        else:
            positions = torch.arange(combined.shape[1], device=combined.device)
            hidden = self.decoder(combined + self.pos_embed(positions).unsqueeze(0), pad_mask=mask)
        return hidden, mask, num_prefix, self._captured.get("h")
