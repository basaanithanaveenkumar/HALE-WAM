"""Dual-system VLA: presets, latents, training recipes, asynchronous control, train.py wiring."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from config import HaloVLMConfig  # noqa: E402
from dual_system import (  # noqa: E402
    PRESETS,
    DualSystemController,
    DualSystemVLA,
    resolve_dual_system_config,
)
from dual_system.resolution import parse_overrides  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402

from tests.support.llm_support import batch_for, hf_patched, small_config  # noqa: E402,F401

SMALL_S1 = {"system1_dim": 32, "system1_layers": 1, "system1_heads": 4, "sample_steps": 2}


def build(name: str, llm: str | None = None, **over) -> DualSystemVLA:
    torch.manual_seed(0)
    cfg = small_config(llm_backbone=llm) if llm else small_config()
    return DualSystemVLA(HaloVLM(cfg), resolve_dual_system_config(name, {**SMALL_S1, **over}))


def batch(model: DualSystemVLA) -> dict:
    b = batch_for(model.system2.model.config)
    b["image_mask"] = torch.ones(2, 2)
    return b


# ---------------------------------------------------------------------------- configuration
def test_none_disables_and_presets_resolve() -> None:
    assert resolve_dual_system_config("none") is None and resolve_dual_system_config(None) is None
    assert set(PRESETS) == {"basic", "lcb", "dp_vla", "hirt", "robodual", "dexvla", "openhelix"}
    cfg = resolve_dual_system_config("openhelix", {"async_steps": 60})
    assert (cfg.system2_training, cfg.aux_weight, cfg.async_steps) == ("prompt_tuning", 1.0, 60)
    with pytest.raises(ValueError, match="unknown dual_system"):
        resolve_dual_system_config("helix")
    with pytest.raises(ValueError, match="unknown dual_system options"):
        resolve_dual_system_config("basic", {"latnt": "pooled"})
    with pytest.raises(ValueError, match="prompt_tuning"):
        resolve_dual_system_config("custom", {"system2_training": "prompt_tuning", "latent": "pooled"})
    with pytest.raises(ValueError):
        resolve_dual_system_config("none", {"latent": "pooled"})


def test_cli_overrides_are_typed() -> None:
    over = parse_overrides(["latent=pooled", "async_steps=4", "aux_weight=0.5", "obs_resampler_tokens=none",
                            "multi_act_with_language=true"])
    assert over == {"latent": "pooled", "async_steps": 4, "aux_weight": 0.5, "obs_resampler_tokens": None,
                    "multi_act_with_language": True}
    with pytest.raises(ValueError):
        parse_overrides(["nonsense"])


# ---------------------------------------------------------------------------- latents
@pytest.mark.parametrize("llm", [None, "smollm2-135m"])
def test_latent_shapes(llm, request) -> None:
    if llm:
        request.getfixturevalue("hf_patched")
    for name, over, k in [
        ("basic", {}, 1),
        ("robodual", {}, 4 + 11),
        ("dp_vla", {}, None),
        ("hirt", {}, 1),
        ("dexvla", {}, 11),
        ("custom", {"latent": "middle_layer"}, None),
    ]:
        model = build(name, llm, **over)
        _, _, enc = model.encode_latent(batch(model))
        tokens = enc["latent_tokens"]
        total = enc["hidden"].size(1)
        assert tokens.size(1) == (total if k is None else k), (name, tokens.shape)
        width = model.system2.model.layer_width if name == "custom" else model.system2.model.config.emb_dim
        assert tokens.size(-1) == width


def test_act_latent_sees_the_image() -> None:
    model = build("basic").eval()
    b = batch(model)
    with torch.no_grad():
        a = model.encode_latent(b)[2]["latent_tokens"]
        b2 = dict(b, images=torch.rand_like(b["images"]))
        c = model.encode_latent(b2)[2]["latent_tokens"]
    assert not torch.allclose(a, c)  # the <ACT> query attends to the visual tokens


# ---------------------------------------------------------------------------- training recipes
def trainable_groups(model: DualSystemVLA) -> set[str]:
    names = {n.split(".")[0] if not n.startswith("system2.") else ("system2.act_queries" if "act_queries" in n
             else "system2.model") for n, p in model.named_parameters() if p.requires_grad}
    return names


@pytest.mark.parametrize("name, expected", [
    ("dp_vla", {"bridge", "system1"}),
    ("openhelix", {"bridge", "system1", "system2.act_queries"}),
    ("basic", {"bridge", "system1", "system2.act_queries", "system2.model"}),
    ("hirt", {"bridge", "system1", "latent", "system2.model"}),
])
def test_system2_training_modes(name: str, expected: set[str]) -> None:
    model = build(name)
    assert trainable_groups(model) == expected
    halo = model.system2.model
    assert not any(p.requires_grad for p in halo.flow_decoder.parameters())  # replaced by System 1


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_trains(name: str) -> None:
    model = build(name).train()
    losses = model(batch(model))
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(p.grad is not None for p in model.system1.parameters())
    if model.config.aux_weight > 0:
        assert losses.aux > 0
    if model.config.align_weight > 0:
        assert 0 <= losses.align <= 2


def test_prompt_tuning_only_moves_the_act_query() -> None:
    model = build("openhelix").train()
    model(batch(model)).total.backward()
    assert model.system2.act_queries.grad is not None
    assert all(p.grad is None for p in model.system2.model.parameters())


def test_prealignment_holds_system1_then_releases_it() -> None:
    model = build("openhelix", prealign_steps=2).train()
    model.set_training_step(0)
    model(batch(model)).total.backward()
    model.mask_frozen_gradients()
    assert all(p.grad is None for p in model.system1.parameters())
    assert model.bridge.projector.weight.grad is not None and model.system2.act_queries.grad is not None
    model.zero_grad(set_to_none=True)
    model.set_training_step(2)
    model(batch(model)).total.backward()
    model.mask_frozen_gradients()
    assert any(p.grad is not None for p in model.system1.parameters())


def test_frame_delay_gives_system2_an_older_view() -> None:
    model = build("hirt")
    images = torch.arange(3.0).view(1, 3, 1, 1, 1).expand(1, 3, 3, 4, 4)
    seen = model.system2.view_images(images)
    assert seen[0, :, 0, 0, 0].tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("name", ["basic", "dp_vla", "lcb"])  # flow matching, regression, diffusion
def test_policy_heads_fit_a_batch(name: str) -> None:
    model = build(name).train()
    b = batch(model)

    def eval_loss() -> float:
        # Generative losses sample noise / timesteps: average over a fixed set of draws.
        torch.manual_seed(123)
        with torch.no_grad():
            return sum(model(b).policy.item() for _ in range(8)) / 8

    before = eval_loss()
    opt = torch.optim.AdamW(model.optimizer_parameters(), lr=3e-3)
    for _ in range(60):
        loss = model(b).policy
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = eval_loss()
    assert after < 0.6 * before, (name, before, after)
    assert model.predict_actions(b).shape == (2, model.chunk, model.action_dim)


@pytest.mark.parametrize("name", ["basic", "dp_vla", "lcb"])
def test_sampled_actions_stay_on_the_action_scale(name: str) -> None:
    """Untrained or not, sampling must not blow up (eps-prediction DDIM amplified errors ~300x)."""
    model = build(name, sample_steps=4).eval()
    actions = model.predict_actions(batch(model))
    assert actions.abs().max() < 20, actions.abs().max()


def test_diffusion_policy_learns_to_sample_the_target() -> None:
    model = build("lcb", sample_steps=8).train()
    b = batch(model)
    b["actions"] = b["actions"].clamp(-2, 2)
    opt = torch.optim.AdamW(model.optimizer_parameters(), lr=3e-3)
    model.eval()
    torch.manual_seed(1)
    before = (model.predict_actions(b) - b["actions"][:, :model.chunk]).abs().mean()
    model.train()
    for _ in range(150):
        loss = model(b).policy
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    torch.manual_seed(1)
    after = (model.predict_actions(b) - b["actions"][:, :model.chunk]).abs().mean()
    assert after < 0.5 * before, (before, after)


# ---------------------------------------------------------------------------- asynchronous control
def test_controller_runs_system2_every_async_steps() -> None:
    model = build("basic", async_steps=3)
    ctrl = DualSystemController(model)
    b = batch(model)
    for _ in range(7):
        actions = ctrl.act(b)
    assert ctrl.system2_calls == 3 and actions.shape == (2, model.chunk, model.action_dim)  # steps 0, 3, 6
    ctrl.reset()
    assert ctrl.latent is None


def test_threaded_controller_never_blocks_after_the_first_latent() -> None:
    model = build("basic", async_steps=2)
    ctrl = DualSystemController(model, threaded=True)
    b = batch(model)
    for _ in range(5):
        assert ctrl.act(b).shape == (2, model.chunk, model.action_dim)
    ctrl.close()
    assert ctrl.system2_calls >= 1
