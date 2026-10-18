"""The export graphs compute exactly what the training-time modules compute."""

from __future__ import annotations

import pytest
import torch
from tests.support.deploy_support import prompt_batch, tiny_dual

from Halo_VLA.deploy.graphs import (
    DenseMoE,
    DualSystemPolicyGraph,
    FlowPolicyGraph,
    MoEDensifier,
    PerceptionGraph,
    System1PolicyGraph,
    System2LatentGraph,
    TokenStateSelector,
)

ATOL = 2e-6


@torch.no_grad()
def test_perception_matches_halo_vlm_forward(tiny_vlm) -> None:
    batch = prompt_batch(tiny_vlm.config)
    _, action_hiddens, _, _ = tiny_vlm(batch["images"], batch["input_ids"], batch["attention_mask"], batch["states"])
    torch.testing.assert_close(PerceptionGraph(tiny_vlm)(**batch), action_hiddens, atol=ATOL, rtol=0)


@torch.no_grad()
def test_flow_policy_matches_sample_actions_with_the_same_noise(tiny_vlm) -> None:
    batch = prompt_batch(tiny_vlm.config)
    _, action_hiddens, _, _ = tiny_vlm(batch["images"], batch["input_ids"], batch["attention_mask"], batch["states"])
    torch.manual_seed(3)
    expected = tiny_vlm.sample_actions(action_hiddens, num_steps=4)
    torch.manual_seed(3)
    noise = torch.randn(2, 1, tiny_vlm.action_chunk_size * tiny_vlm.action_dim)
    got = FlowPolicyGraph(tiny_vlm, 1, num_steps=4)(**batch, noise=noise)
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("preset,over", [
    ("basic", {}), ("lcb", {}), ("dp_vla", {}), ("hirt", {}), ("robodual", {}), ("dexvla", {}), ("openhelix", {}),
    ("custom", {"latent": "middle_layer"}),
])
@torch.no_grad()
def test_dual_system_graphs_match_predict_actions(preset, over) -> None:
    dual = tiny_dual(preset, **over)
    batch = prompt_batch(dual.system2.model.config)
    torch.manual_seed(7)
    expected = dual.predict_actions(batch)
    torch.manual_seed(7)
    noise = torch.randn(2, dual.chunk, dual.action_dim)
    torch.testing.assert_close(DualSystemPolicyGraph(dual)(**batch, noise=noise), expected, atol=ATOL, rtol=0)
    # the split graphs compose to the same thing (System 2 -> latent -> System 1)
    latent, mask = System2LatentGraph(dual)(**batch)
    ref_latent, ref_mask, _ = dual.encode_latent(batch)
    torch.testing.assert_close(latent, ref_latent, atol=ATOL, rtol=0)
    assert torch.equal(mask, ref_mask)
    got = System1PolicyGraph(dual)(latent, mask, batch["images"][:, -1], batch["states"][:, -1], noise)
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


def test_token_selector_picks_occurrences_in_order_and_zero_pads() -> None:
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    ids = torch.tensor([[1, 9, 1, 9, 9], [1, 1, 9, 1, 1]])
    out = TokenStateSelector(token_id=9, count=3)(hidden, ids)
    assert torch.equal(out[0], hidden[0, [1, 3, 4]])
    assert torch.equal(out[1, 0], hidden[1, 2]) and torch.equal(out[1, 1:], torch.zeros(2, 3))
    with pytest.raises(ValueError):
        TokenStateSelector(9, 0)


@torch.no_grad()
def test_dense_moe_matches_sparse_dispatch_and_densifier_swaps_layers() -> None:
    from config import HaloVLMConfig
    from models.halo_vla import HaloVLM
    from models.moe import DeepseekMoE

    torch.manual_seed(0)
    moe = DeepseekMoE(32, 40, 6, 2, 2).eval()
    x = torch.randn(3, 7, 32)
    torch.testing.assert_close(DenseMoE(moe)(x), moe(x), atol=1e-6, rtol=0)
    from tests.support.deploy_support import TINY_MODEL

    vlm = HaloVLM(HaloVLMConfig(**{**TINY_MODEL, "use_moe": True, "moe_num_routed_experts": 4,
                                    "moe_num_shared_experts": 1})).eval()
    batch = prompt_batch(vlm.config)
    before = PerceptionGraph(vlm)(**batch)
    assert MoEDensifier().apply(vlm) == 2 and MoEDensifier().apply(vlm) == 0  # idempotent
    torch.testing.assert_close(PerceptionGraph(vlm)(**batch), before, atol=ATOL, rtol=0)
