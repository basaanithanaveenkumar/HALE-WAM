"""``model.llm``: the language decoder - built-in, or any fine-tunable LLM from HaleBlocks.

Same fields and semantics as Hale-VLM's ``model.llm`` section, plus ``backbone: null`` for
Halo-VLA's own small decoder.
"""

from __future__ import annotations

from typing import Any

from pydantic import field_validator

from config.sections.base import VLASection
from exceptions import HaloVLAConfigError


class LLMConfig(VLASection):
    """Which LLM is the VLA's decoder and how it is fine-tuned.

    ``backbone``: ``null`` keeps the built-in decoder (cosmo2 tokenizer); otherwise any name
    from ``hale_core.nn.llm.list_llm_backbones()`` - text LLMs (``qwen3-1.7b``, ``smollm3-3b``,
    ``granite4.2-3b`` ...) or text towers of natively multimodal models (``qwen3.5-4b-base``,
    ``gemma4-e4b``, ``ministral3-3b``, ``cosmos-reason2-2b``, ``lfm2.5-vl-1.6b``,
    ``smolvlm2-500m`` ...) - or ``custom`` with ``model_id``. Fine-tuning, as in Hale-VLM:
    ``use_lora`` (adapters) > ``freeze_llm`` (no LLM gradients) > full fine-tuning.
    """

    ARGS = {"backbone": "llm_backbone", "model_id": "llm_model_id", "checkpoint": "llm_checkpoint",
            "dtype": "llm_dtype", "lora_r": "llm_lora_r", "lora_alpha": "llm_lora_alpha",
            "lora_dropout": "llm_lora_dropout", "gradient_checkpointing": "llm_gradient_checkpointing",
            "attn_implementation": "llm_attn_implementation"}
    OVERRIDES = {"lora_target_modules": "llm_lora_target_modules"}

    backbone: str | None = None
    model_id: str | None = None            # another repo / local path of the same family
    checkpoint: str | None = None          # HaloDiffusionLLM .pt for hale-* backbones
    dtype: str = "bfloat16"
    freeze_llm: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    lora_train_embeddings: bool = True     # the VLA's new special tokens must be learned
    gradient_checkpointing: bool = False
    attn_implementation: str | None = "sdpa"   # Gemma: eager

    @field_validator("backbone")
    @classmethod
    def _known_backbone(cls, name: str | None) -> str | None:
        if name in (None, "custom"):
            return name
        from hale_core.nn.llm import is_registered, list_llm_backbones

        if not is_registered(name):
            raise HaloVLAConfigError(f"unknown LLM backbone {name!r}; choose one of {list_llm_backbones()} or 'custom'")
        return name

    @field_validator("dtype")
    @classmethod
    def _known_dtype(cls, dtype: str) -> str:
        if dtype not in ("float32", "float16", "bfloat16"):
            raise HaloVLAConfigError(f"dtype must be float32, float16 or bfloat16, got {dtype!r}")
        return dtype

    @property
    def finetune(self) -> str:
        """``lora`` / ``frozen`` / ``full`` from the two flags (LoRA wins)."""
        return "lora" if self.use_lora else ("frozen" if self.freeze_llm else "full")

    def train_arguments(self) -> dict[str, Any]:
        out = super().train_arguments()
        given = self.model_fields_set
        if given & {"use_lora", "freeze_llm"}:
            out["llm_finetune"] = self.finetune
        if "lora_train_embeddings" in given:
            out["llm_freeze_embeddings"] = not self.lora_train_embeddings
        return out
