"""Q-Former / gated cross-attention image projectors for HaloVLM."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config.model_config import HaloVLMConfig
from models.gated_cross_attention import GatedCrossAttentionProjector
from models.halo_vla import HaloVLM
from models.image_proj import ImageProjector, build_image_projector
from models.qformer import QFormer

VISION_DIM, LLM_DIM = 32, 16


def _small_config(**overrides) -> HaloVLMConfig:
    cfg = HaloVLMConfig()
    cfg.img_size = 32
    cfg.patch_size = 16  # 2x2 = 4 ViT patches per image
    cfg.emb_dim = 64
    cfg.vit_num_layers = 1
    cfg.dec_num_layers = 2
    cfg.dit_patch_size = 4
    cfg.dit_hidden_size = 64
    cfg.dit_depth = 2
    cfg.dit_num_heads = 2
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_qformer_shapes_and_cross_attention_frequency():
    model = QFormer(
        VISION_DIM, LLM_DIM, num_queries=8, hidden_dim=16, num_layers=4, num_heads=4,
        cross_attention_freq=2,
    )
    assert [layer.has_cross_attention for layer in model.layers] == [True, False, True, False]
    assert model(torch.randn(2, 49, VISION_DIM)).shape == (2, 8, LLM_DIM)
    assert model(torch.randn(2, 196, VISION_DIM)).shape == (2, 8, LLM_DIM)


def test_qformer_depends_on_image_and_trains_every_parameter():
    model = QFormer(VISION_DIM, LLM_DIM, num_queries=4, hidden_dim=16, num_heads=4)
    a, b = torch.randn(2, 9, VISION_DIM), torch.randn(2, 9, VISION_DIM)
    assert not torch.allclose(model(a), model(b))
    model(a).square().mean().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_gated_cross_attention_gate_controls_the_image_path():
    kwargs = dict(num_latents=6, hidden_dim=16, num_layers=2, num_heads=4)
    a, b = torch.randn(2, 9, VISION_DIM), torch.randn(2, 9, VISION_DIM)

    closed = GatedCrossAttentionProjector(VISION_DIM, LLM_DIM, gate_init=0.0, **kwargs)
    assert closed(a).shape == (2, 6, LLM_DIM)
    assert torch.allclose(closed(a), closed(b)), "tanh(0) gate must hide the image"
    closed(a).square().mean().backward()
    assert all(block.attn_gate.grad.abs().item() > 0 for block in closed.blocks)

    opened = GatedCrossAttentionProjector(VISION_DIM, LLM_DIM, gate_init=1.0, **kwargs)
    assert not torch.allclose(opened(a), opened(b))
    opened(a).square().mean().backward()
    assert [n for n, p in opened.named_parameters() if p.grad is None] == []


@pytest.mark.parametrize("cls", [QFormer, GatedCrossAttentionProjector])
def test_heads_must_divide_hidden_dim(cls):
    with pytest.raises(ValueError, match="divisible"):
        cls(VISION_DIM, LLM_DIM, hidden_dim=10, num_heads=4)


def test_config_validates_and_counts_image_tokens():
    assert HaloVLMConfig().proj_type == "mlp"
    assert HaloVLMConfig().num_image_tokens_per_image == (224 // 16) ** 2
    assert HaloVLMConfig(proj_type="qformer", qformer_num_queries=12).num_image_tokens_per_image == 12
    gated = HaloVLMConfig(proj_type="gated_cross_attention", gated_xattn_num_latents=7)
    assert gated.num_image_tokens_per_image == 7
    with pytest.raises(ValueError, match="proj_type"):
        HaloVLMConfig(proj_type="perceiver")


def test_default_projector_is_unchanged():
    cfg = _small_config()
    projector = build_image_projector(cfg)
    assert type(projector) is ImageProjector
    keys = HaloVLM(cfg).state_dict().keys()
    assert "image_projector.proj.0.weight" in keys, "old checkpoints must still load"


@pytest.mark.parametrize(
    ("proj_type", "extra", "cls", "tokens"),
    [
        ("qformer", {"qformer_num_queries": 3, "qformer_num_heads": 4}, QFormer, 3),
        (
            "gated_cross_attention",
            {"gated_xattn_num_latents": 5, "gated_xattn_num_heads": 4},
            GatedCrossAttentionProjector,
            5,
        ),
    ],
)
def test_halo_vlm_forward_and_backward_with_connector(proj_type, extra, cls, tokens):
    cfg = _small_config(proj_type=proj_type, **extra)
    model = HaloVLM(cfg)
    assert isinstance(model.image_projector, cls)
    assert cfg.num_image_tokens_per_image == tokens

    batch, seq_len, n_img = 2, 6, 2
    images = torch.randn(batch, n_img, 3, 32, 32)
    input_ids = torch.randint(0, 100, (batch, seq_len))
    input_ids[:, :n_img] = cfg.image_token_id
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    states = torch.zeros(batch, 1, cfg.state_dim)

    logits, _, visual_context, _ = model(
        images, input_ids, attention_mask, states, image_mask=torch.ones(batch, n_img)
    )
    assert logits.shape == (batch, n_img * tokens + seq_len, cfg.vocab_size)
    assert visual_context.shape == (batch, cfg.emb_dim)

    logits.mean().backward()
    assert all(p.grad is not None for p in model.image_projector.parameters())
    assert model.encode_visual_context_from_past_vit(images).shape == (batch, cfg.emb_dim)
