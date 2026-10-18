"""The RL package (REINFORCE / GRPO / DPO / Dreamer) against the current HaloVLM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from models.halo_vla import HaloVLM  # noqa: E402
from rl import (  # noqa: E402
    RSSM,
    GaussianActionPolicy,
    compute_dpo_loss,
    compute_grpo_loss,
    compute_reinforce_loss,
    world_model_loss,
)
from tests.support.llm_support import batch_for, hf_patched, small_config  # noqa: E402,F401


def test_every_algorithm_is_exported() -> None:
    assert all(
        callable(f)
        for f in (compute_dpo_loss, compute_grpo_loss, compute_reinforce_loss, world_model_loss)
    )
    assert RSSM is not None


@pytest.mark.parametrize("llm", [None, "smollm2-135m"])
def test_policy_conditioning_from_the_current_model(llm, request) -> None:
    if llm is not None:
        request.getfixturevalue("hf_patched")
    cfg = small_config(llm_backbone=llm) if llm else small_config()
    model = HaloVLM(cfg).eval()
    cond = GaussianActionPolicy.conditioning_from_halo(model, batch_for(cfg))
    assert cond.shape == (2, cfg.emb_dim) and torch.isfinite(cond).all()
    policy = GaussianActionPolicy(
        cond_dim=cfg.emb_dim, action_dim_flat=cfg.action_chunk_size * cfg.action_dim
    )
    actions, log_prob = policy.sample(cond)
    assert actions.shape == (2, cfg.action_chunk_size * cfg.action_dim) and log_prob.shape == (2,)


def test_dreamer_frame_encoding_uses_the_vision_tower() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_dreamer import encode_frames

    cfg = small_config()
    model = HaloVLM(cfg).eval()
    with torch.no_grad():
        obs = encode_frames(model, torch.rand(2, 3, 3, cfg.img_size, cfg.img_size))
    assert obs.shape == (2, 3, cfg.emb_dim)
