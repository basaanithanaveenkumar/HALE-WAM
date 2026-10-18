"""(d) The VLA's pretrained LLM as a Hugging Face checkpoint (TensorRT-LLM, mlx-lm, vLLM)."""

from __future__ import annotations

from typing import Any

from loguru import logger
from torch import nn

from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import ConfigError
from Halo_VLA.deploy.adapters.halo_adapter_base import HaloAdapterBase
from Halo_VLA.deploy.graphs.text_tower_graph import TextTowerGraph
from Halo_VLA.deploy.registry import HALO_VLA_ADAPTERS


@HALO_VLA_ADAPTERS.register_module()
class HaloLanguageModelAdapter(HaloAdapterBase):
    """System 2's language tower (``model.llm_backbone``) for LLM serving stacks.

    :meth:`hf_checkpoint` returns the ``*ForCausalLM`` with LoRA adapters **merged** and the
    VLA's special tokens in its vocabulary/tokenizer, ready for ``HuggingFaceCheckpointStage``.
    It covers the text path (instructions, reasoning); the vision prefix is spliced in by the
    VLA's own bridges, which LLM servers only support through custom-embedding inputs.

    The ONNX/MLX graph (``input_ids, attention_mask -> logits``) exists for parity checks of the
    text tower.

    Args:
        max_seq: largest text length of the exported graph.
    """

    def __init__(self, max_seq: int = 512, opt_seq: int = 32, **base: Any) -> None:
        super().__init__(**base)
        if not self.factory.model_kwargs.get("llm_backbone"):
            raise ConfigError("HaloLanguageModelAdapter needs model.llm_backbone (a pretrained HF LLM); the "
                              "built-in decoder has no Hugging Face form")
        self.max_seq, self.opt_seq = int(max_seq), int(opt_seq)

    def language(self, model: nn.Module) -> nn.Module:
        """The pretrained-LLM wrapper inside ``model``."""
        return self.vlm(model).language

    def default_signature(self) -> IOSignature:
        """Token ids and mask in; next-token logits out."""
        cfg = self.model_config()
        return self.build_signature({
            "inputs": [{"name": "input_ids", "shape": ["batch", "seq"], "dtype": "int64",
                        "value_range": [3, cfg.vocab_size]},
                       {"name": "attention_mask", "shape": ["batch", "seq"], "dtype": "int64", "value_range": [1, 2]}],
            "outputs": ["logits"],
            "dims": {"batch": self.batch_dim(), "seq": [1, self.opt_seq, self.max_seq]},
        })

    def graph(self, model: nn.Module) -> nn.Module:
        """The text tower (embedding, LLM, LM head) without the vision prefix."""
        return TextTowerGraph(self.language(model))

    def hf_checkpoint(self, model: nn.Module) -> tuple[Any, Any]:
        """The LLM as a Hugging Face model with LoRA merged, plus its tokenizer.

        Args:
            model: the built VLA.

        Returns:
            ``(hf_model, tokenizer)`` for ``HuggingFaceCheckpointStage``.

        Raises:
            ConfigError: if the backbone is not a Hugging Face model.
        """
        backbone = self.language(model).llm
        hf = getattr(backbone, "model", None)
        if hf is None:
            raise ConfigError(f"LLM backbone {type(backbone).__name__} is not a Hugging Face model")
        if hasattr(hf, "merge_and_unload"):
            logger.info("merging LoRA adapters into the base LLM for deployment")
            hf = hf.merge_and_unload()
        return hf, backbone.tokenizer
