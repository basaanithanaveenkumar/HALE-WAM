"""Import roots, tiny-model settings and helpers shared by the deployment tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src" / "Halo_VLA", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CONFIGS = ROOT / "configs" / "deploy"
#: HaloVLMConfig of the CI models (matches configs/deploy/_shared/models/tiny_*.yaml).
TINY_MODEL = dict(
    img_size=32,
    patch_size=16,
    emb_dim=64,
    vit_num_layers=1,
    vit_num_heads=4,
    vit_mlp_dim=64,
    dec_num_layers=2,
    dec_num_heads=4,
    dec_mlp_dim=64,
    use_moe=False,
    action_dim=4,
    state_dim=3,
    action_chunk_size=2,
    enable_visual_dit=False,
    flow_hidden_dim=32,
    flow_num_ode_steps=4,
)
TINY_DUAL = {"system1_dim": 32, "system1_layers": 1, "system1_heads": 4, "sample_steps": 3}
TINY_PROMPT = {
    "num_images": 2,
    "num_states": 1,
    "num_action_tokens": 1,
    "num_world_video_tokens": 1,
    "opt_seq": 16,
    "max_seq": 48,
}


def prompt_batch(cfg, b: int = 2, text: int = 6) -> dict:
    """Template-shaped observation (two frames, one state, one world-video and one action token)."""
    words = torch.randint(3, 40, (b, text))
    special = torch.tensor(
        [
            cfg.image_token_id,
            cfg.image_token_id,
            cfg.state_token_id,
            cfg.world_video_token_id,
            cfg.action_token_id,
        ]
    ).expand(b, -1)
    ids = torch.cat([special[:, :3], words, special[:, 3:]], 1)
    return {
        "images": torch.rand(b, 2, 3, cfg.img_size, cfg.img_size),
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "states": torch.randn(b, 1, cfg.state_dim),
    }


def tiny_dual(preset: str, **over):
    from config import HaloVLMConfig
    from dual_system import DualSystemVLA, resolve_dual_system_config
    from models.halo_vla import HaloVLM

    torch.manual_seed(0)
    return DualSystemVLA(
        HaloVLM(HaloVLMConfig(**TINY_MODEL)),
        resolve_dual_system_config(preset, {**TINY_DUAL, **over}),
    ).eval()
