"""Text-only forward of the VLA's pretrained LLM (System 2 language reasoning)."""

from __future__ import annotations

import torch
from torch import nn


class TextTowerGraph(nn.Module):
    """``input_ids, attention_mask -> logits`` through the registered LLM backbone.

    The VLA bridges (``to_llm`` / ``from_llm``) and the vision prefix are not involved: this is
    the language half that TensorRT-LLM / mlx-lm serve from the exported HF checkpoint.

    Args:
        language: the VLA's ``PretrainedLanguageDecoder``.
    """

    def __init__(self, language: nn.Module) -> None:
        super().__init__()
        self.language = language

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.language.embed_text(input_ids)
        logits, _hidden = self.language(embeds, attention_mask, input_ids)
        return logits
