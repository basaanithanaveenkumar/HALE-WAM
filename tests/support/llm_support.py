"""Offline stand-ins for a pretrained LLM backbone, shared by the LLM, dual-system and RL tests.

The Hugging Face loaders are patched to return a tiny random Llama-architecture model and an
in-memory word-level tokenizer, so nothing is downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("hale_core")

from config import HaloVLMConfig  # noqa: E402


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(["<pad>", "<unk>", "<eos>", *(f"w{i}" for i in range(40))])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
                                                eos_token="<eos>")


def tiny_llama(hidden: int = 48):
    cfg = transformers.LlamaConfig(vocab_size=43, hidden_size=hidden, intermediate_size=96, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                                   pad_token_id=0, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg)


def small_config(**llm) -> HaloVLMConfig:
    cfg = HaloVLMConfig(img_size=32, patch_size=16, emb_dim=64, vit_num_layers=1, vit_num_heads=4, vit_mlp_dim=64,
                        dec_num_layers=1, dec_num_heads=4, dec_mlp_dim=64, use_moe=False, action_dim=4,
                        state_dim=3, action_chunk_size=2, enable_visual_dit=False, flow_hidden_dim=32,
                        llm_dtype="float32", **llm)
    return cfg


def batch_for(cfg: HaloVLMConfig, b: int = 2) -> dict:
    """Two images, one state, one action and one world-video token per sample."""
    text = torch.randint(3, 40, (b, 6))
    special = torch.tensor([cfg.image_token_id, cfg.image_token_id, cfg.state_token_id,
                            cfg.action_token_id, cfg.world_video_token_id]).expand(b, -1)
    ids = torch.cat([special[:, :3], text, special[:, 3:]], 1)
    return {
        "images": torch.rand(b, 2, 3, cfg.img_size, cfg.img_size),
        "input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone(),
        "states": torch.randn(b, 1, cfg.state_dim),
        "actions": torch.randn(b, cfg.action_chunk_size, cfg.action_dim),
        "action_mask": torch.ones(b, cfg.action_chunk_size),
    }


@pytest.fixture
def hf_patched():
    with patch.object(transformers.AutoModelForCausalLM, "from_pretrained", side_effect=lambda *a, **k: tiny_llama()), \
         patch.object(transformers.AutoTokenizer, "from_pretrained", side_effect=lambda *a, **k: tiny_tokenizer()):
        yield
