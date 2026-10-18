"""Adapters: registry scope, signatures, prompt-shaped samples, checkpoints, HF export."""

from __future__ import annotations

import pytest
import torch
from hale_core.deploy.exceptions import ConfigError
from hale_core.deploy.registry import ADAPTERS

from Halo_VLA.deploy import HALO_VLA_ADAPTERS, PromptIOSignature
from Halo_VLA.deploy.adapters import (
    DualSystem1ActionStepAdapter,
    DualSystem1ObserverAdapter,
    DualSystem1PolicyAdapter,
    DualSystem2EncoderAdapter,
    HaloFlowActionStepAdapter,
    HaloLanguageModelAdapter,
    HaloPerceptionAdapter,
)
from tests.support.deploy_support import TINY_DUAL, TINY_MODEL, TINY_PROMPT

SINGLE = {"model": TINY_MODEL, "max_batch": 2}
DUAL = {
    "model": TINY_MODEL,
    "dual_system": "basic",
    "dual_system_options": TINY_DUAL,
    "max_batch": 2,
}


def run(adapter, which: str = "trace"):
    model = adapter.load()
    module = adapter.export_module(model)
    sig = adapter.signature()
    with torch.no_grad():
        return sig, module(*sig.as_args(sig.sample(which, torch.Generator().manual_seed(0))))


def test_child_registry_scope() -> None:
    assert HALO_VLA_ADAPTERS.scope == "halo_vla" and HALO_VLA_ADAPTERS.parent is ADAPTERS
    assert ADAPTERS.get("halo_vla.HaloPerceptionAdapter") is HaloPerceptionAdapter
    assert HALO_VLA_ADAPTERS.get("TorchModuleAdapter") is ADAPTERS.get(
        "TorchModuleAdapter"
    )  # parent fallback
    assert len(HALO_VLA_ADAPTERS) == 9
    built = ADAPTERS.build({"type": "halo_vla.HaloFlowActionStepAdapter", **SINGLE})
    assert isinstance(built, HaloFlowActionStepAdapter)


def test_perception_signature_samples_follow_the_prompt_template() -> None:
    adapter = HaloPerceptionAdapter(**SINGLE, **TINY_PROMPT)
    sig = adapter.signature()
    assert isinstance(sig, PromptIOSignature)
    assert sig.input_names == ["images", "input_ids", "attention_mask", "states"]
    cfg = adapter.model_config()
    ids = sig.sample("min")["input_ids"]
    assert ids.shape[1] == sig.dims["seq"].min == sig.min_prompt_length
    row = ids[0].tolist()
    assert row[1:4] == [cfg.image_token_id, cfg.image_token_id, cfg.state_token_id]
    assert row[-2:] == [cfg.world_video_token_id, cfg.action_token_id]
    _, (context,) = run(adapter)
    assert (
        context.shape == (2, 1, cfg.emb_dim) and context.abs().sum() > 0
    )  # the action token was found


def test_step_and_policy_shapes() -> None:
    sig, (velocity,) = run(HaloFlowActionStepAdapter(**SINGLE))
    assert velocity.shape == (2, TINY_MODEL["action_chunk_size"] * TINY_MODEL["action_dim"])
    _, (latent, mask) = run(DualSystem2EncoderAdapter(**DUAL, **TINY_PROMPT))
    assert latent.shape == (2, 1, TINY_DUAL["system1_dim"]) and mask.shape == (2, 1)
    observer = DualSystem1ObserverAdapter(**DUAL)
    _, (condition, condition_mask) = run(observer)
    assert condition.shape[1] == 1 + observer.observation_tokens() + 1
    step = DualSystem1ActionStepAdapter(**DUAL)
    assert step.signature().dims["context"].opt == condition.shape[1]
    _, (prediction,) = run(step)
    assert prediction.shape == (2, TINY_MODEL["action_chunk_size"], TINY_MODEL["action_dim"])
    _, (actions,) = run(DualSystem1PolicyAdapter(**DUAL))
    assert actions.shape == prediction.shape


def test_regression_policy_has_no_noise_input_and_no_step() -> None:
    reg = {**DUAL, "dual_system": "dp_vla"}
    assert (
        "noise"
        not in DualSystem1PolicyAdapter(**reg, latent_tokens=[1, 4, 16]).signature().input_names
    )
    with pytest.raises(ConfigError, match="regression"):
        DualSystem1ActionStepAdapter(**reg).signature()


def test_single_vs_dual_guards_and_bad_model_fields() -> None:
    with pytest.raises(ConfigError, match="DualSystem"):
        HaloPerceptionAdapter(**DUAL)
    with pytest.raises(ConfigError, match="dual_system"):
        DualSystem2EncoderAdapter(**SINGLE)
    with pytest.raises(ConfigError, match="HaloVLMConfig"):
        HaloPerceptionAdapter(model={"not_a_field": 1}).load()
    with pytest.raises(ConfigError, match="invalid dual-system"):
        DualSystem2EncoderAdapter(**{**DUAL, "dual_system": "nope"}).load()
    with pytest.raises(ConfigError, match="llm_backbone"):
        HaloLanguageModelAdapter(**SINGLE)


def test_training_checkpoint_loads_into_the_full_model(tmp_path) -> None:
    adapter = HaloFlowActionStepAdapter(**SINGLE)
    model = adapter.build_model()
    torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, tmp_path / "ckpt.pt")
    fresh = HaloFlowActionStepAdapter(**SINGLE, seed=99, checkpoint=str(tmp_path / "ckpt.pt"))
    loaded = fresh.load()
    assert torch.equal(loaded.flow_decoder.net[0].weight, model.flow_decoder.net[0].weight)


def test_calibration_batches_are_template_shaped() -> None:
    adapter = HaloPerceptionAdapter(**SINGLE, **TINY_PROMPT)
    batches = list(adapter.calibration_batches(3))
    cfg = adapter.model_config()
    assert len(batches) == 3
    assert all(bool(((b["input_ids"] == cfg.action_token_id).sum(1) == 1).all()) for b in batches)
    assert not torch.equal(batches[0]["images"], batches[1]["images"])


def test_language_model_adapter_exports_a_merged_hf_checkpoint(monkeypatch) -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from unittest.mock import patch

    from tests.support.llm_support import tiny_llama, tiny_tokenizer

    with (
        patch.object(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            side_effect=lambda *a, **k: tiny_llama(),
        ),
        patch.object(
            transformers.AutoTokenizer,
            "from_pretrained",
            side_effect=lambda *a, **k: tiny_tokenizer(),
        ),
    ):
        adapter = HaloLanguageModelAdapter(
            model={**TINY_MODEL, "llm_backbone": "smollm2-135m", "llm_dtype": "float32"}
        )
        model = adapter.load()
        hf, tokenizer = adapter.hf_checkpoint(model)
        assert not hasattr(hf, "peft_config")  # LoRA merged
        assert "<halo_action>" in tokenizer.get_vocab()
        cfg = adapter.model_config()
        assert cfg.action_token_id == tokenizer.convert_tokens_to_ids("<halo_action>")
        sig, (logits,) = run(adapter)
        assert logits.shape[-1] == hf.get_input_embeddings().weight.shape[0]
