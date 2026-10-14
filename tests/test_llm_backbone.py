"""Halo-VLA with a pretrained, LoRA-finetunable LLM decoder (``config.llm_backbone``).

Offline: the Hugging Face loaders are patched to return a tiny random Llama-architecture model
and an in-memory word-level tokenizer; the HaloDiffusionLLM path uses a real checkpoint file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("hale_core")

from config import HaloVLMConfig  # noqa: E402
from config.tokens import get_all_custom_tokens  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402
from training import HaloVLATrainingStep  # noqa: E402


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


def test_hf_backbone_replaces_the_decoder_and_rewrites_token_ids(hf_patched) -> None:
    cfg = small_config(llm_backbone="smollm2-135m")
    model = HaloVLM(cfg)
    assert model.decoder_transformer is None and not hasattr(model, "token_emb")
    tok = model.language.tokenizer
    assert [cfg.image_token_id, cfg.action_token_id, cfg.state_token_id, cfg.world_video_token_id] == \
        [tok.convert_tokens_to_ids(t) for t in get_all_custom_tokens()]
    assert cfg.vocab_size == len(tok) == 47
    batch = batch_for(cfg)
    logits, action_hiddens, _, world = model(batch["images"], batch["input_ids"], batch["attention_mask"],
                                             batch["states"])
    prepended = 2 * cfg.num_image_tokens_per_image
    assert logits.shape == (2, prepended + batch["input_ids"].size(1), 47)
    assert action_hiddens.shape == (2, 1, cfg.emb_dim) and world.shape == (2, 1, cfg.emb_dim)


def test_lora_trains_adapters_embeddings_and_bridges_only(hf_patched) -> None:
    cfg = small_config(llm_backbone="qwen2.5-0.5b", llm_finetune="lora", llm_lora_r=4, llm_lora_alpha=8)
    model = HaloVLM(cfg)
    llm_trainable = {n for n, p in model.language.llm.named_parameters() if p.requires_grad}
    assert llm_trainable and all("lora_" in n or "embed_tokens" in n or "lm_head" in n for n in llm_trainable)
    frozen_base = [n for n, p in model.language.llm.named_parameters()
                   if "q_proj.base_layer" in n and p.requires_grad]
    assert not frozen_base
    step = HaloVLATrainingStep(model)
    out = step(batch_for(cfg))
    assert torch.isfinite(out.total)
    out.total.backward()
    assert model.language.to_llm.weight.grad is not None and model.language.from_llm.weight.grad is not None
    assert all(p.grad is not None for n, p in model.language.llm.named_parameters() if "lora_B" in n)


def test_frozen_llm_still_trains_the_vla_heads(hf_patched) -> None:
    cfg = small_config(llm_backbone="smollm2-135m", llm_finetune="frozen")
    model = HaloVLM(cfg)
    assert not any(p.requires_grad for p in model.language.llm.parameters())
    HaloVLATrainingStep(model)(batch_for(cfg)).total.backward()
    assert model.flow_decoder is not None and any(p.grad is not None for p in model.flow_decoder.parameters())


def test_hale_checkpoint_backbone(tmp_path: Path) -> None:
    from hale_core.nn.backbones import build_backbone

    tok = tiny_tokenizer()
    tok.save_pretrained(tmp_path / "tok")
    model_cfg = dict(d_model=32, n_heads=4, n_layers=1, d_ff=64, max_length=64, dropout=0.0, attn_type="causal",
                     use_time_cond=False, block_size=None, attn_impl="gqa", n_kv_heads=2, ffn_type="geglu",
                     moe_num_experts=4, moe_top_k=2, moe_num_shared=1, arch="transformer", sliding_window=None,
                     local_global_ratio=None, rope_theta_local=None, rope_theta_global=None, p_rope=None,
                     qk_norm=False)
    bb = build_backbone(len(tok), SimpleNamespace(**model_cfg))
    torch.save({"variant": "autoregressive", "vocab_size": len(tok),
                "model_state_dict": {f"backbone.{k}": v for k, v in bb.state_dict().items()},
                "config": {"model": model_cfg, "data": {"tokenizer_name": str(tmp_path / "tok")}}},
               tmp_path / "ar.pt")
    cfg = small_config(llm_backbone="hale-autoregressive", llm_checkpoint=str(tmp_path / "ar.pt"),
                       llm_lora_r=4, llm_lora_alpha=8)
    model = HaloVLM(cfg)
    assert model.language.hidden_size == 32
    out = HaloVLATrainingStep(model)(batch_for(cfg))
    out.total.backward()
    assert torch.isfinite(out.total)


def test_built_in_decoder_is_unchanged_without_llm_backbone() -> None:
    cfg = small_config()
    model = HaloVLM(cfg)
    assert model.language is None and model.decoder_transformer is not None
    assert cfg.image_token_id == 49152 and model.token_emb.num_embeddings == cfg.vocab_size
