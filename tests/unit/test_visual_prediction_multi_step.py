import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config.model_config import HaloVLMConfig
from models.dit_frame_prediction import VisualDiTPredictor, make_dit_config
from models.halo_vla import HaloVLM


def test_dit_config_fallbacks():
    class DummyConfig:
        emb_dim = 128
        patch_size = 8
        dit_depth = 4
        dit_num_heads = 4
        dit_mlp_ratio = 2.0
        dit_max_resolution = 64

    cfg = DummyConfig()
    dit_cfg = make_dit_config(cfg, 3)
    assert dit_cfg.dit_patch_size == 8
    assert dit_cfg.dit_hidden_size == 128
    assert dit_cfg.dit_depth == 4
    assert dit_cfg.dit_num_heads == 4
    assert dit_cfg.dit_mlp_ratio == 2.0
    assert dit_cfg.dit_max_resolution == 64
    print("test_dit_config_fallbacks passed.")


def test_visual_dit_predictor_rgb_only_default():
    cfg = HaloVLMConfig()
    cfg.dit_patch_size = 4
    cfg.dit_hidden_size = 64
    cfg.dit_depth = 2
    cfg.dit_num_heads = 2
    cfg.emb_dim = 64

    cfg.dit_max_resolution = 32
    predictor = VisualDiTPredictor(cfg, num_predict_frames=2)

    # Two future RGB frames conditioned on a pooled context embedding.
    B, F, C, H, W = 2, 2, 3, 32, 32
    future_rgb = torch.randn(B, F, C, H, W)
    context_emb = torch.randn(B, cfg.emb_dim)

    # RGB is the only modality: the losses are the RGB flow-matching terms, no depth / flow.
    losses = predictor.compute_loss(future_rgb, context_emb, num_frames=F, perceptual_weight=0.0)
    assert {"loss_rgb_cfm", "loss_rgb_total"} <= set(losses)
    assert not any(k.startswith(("loss_depth", "loss_flow")) for k in losses)
    assert torch.isfinite(losses["loss_rgb_total"]) and losses["loss_rgb_cfm"] > 0
    print("test_visual_dit_predictor_rgb_only_default passed.")


def test_halo_vlm_multi_step_prediction_loss():
    cfg = HaloVLMConfig()
    cfg.img_size = 32
    cfg.patch_size = 16
    cfg.dit_patch_size = 4
    cfg.dit_hidden_size = 64
    cfg.dit_depth = 2
    cfg.dit_num_heads = 2
    cfg.emb_dim = 64
    cfg.vit_num_layers = 1
    cfg.dec_num_layers = 2

    model = HaloVLM(cfg)

    B, N, C, H, W = 2, 2, 3, 32, 32
    images = torch.randn(B, N, C, H, W)
    future_frames = torch.randn(B, 3, C, H, W)
    visual_context_emb = torch.randn(B, cfg.emb_dim)

    # Without future frames there is nothing to predict.
    zero, parts = model.compute_visual_prediction_loss(
        images, visual_context_emb=visual_context_emb
    )
    assert zero.item() == 0.0 and parts == {}
    # With 3 future frames the loss is a finite scalar over the predicted frames.
    loss, parts = model.compute_visual_prediction_loss(
        images,
        visual_context_emb=visual_context_emb,
        future_frames=future_frames,
        perceptual_weight=0.0,
    )
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0 and torch.isfinite(loss) and loss > 0
    assert "loss_rgb_cfm" in parts
    print("test_halo_vlm_multi_step_prediction_loss passed.")


def test_halo_vlm_predict_visual_future_autoregressive():
    cfg = HaloVLMConfig()
    cfg.img_size = 32
    cfg.patch_size = 16
    cfg.dit_patch_size = 4
    cfg.dit_hidden_size = 64
    cfg.dit_depth = 2
    cfg.dit_num_heads = 2
    cfg.emb_dim = 64
    cfg.vit_num_layers = 1
    cfg.dec_num_layers = 2

    model = HaloVLM(cfg)

    B, T, C, H, W = 2, 2, 3, 32, 32
    past_rgb = torch.randn(B, T, C, H, W)
    visual_context_emb = torch.randn(B, cfg.emb_dim)

    # Autoregressively predict 3 frames
    preds = model.predict_visual_future(
        past_rgb, visual_context_emb=visual_context_emb, num_frames=3
    )
    assert "rgb" in preds
    assert "depth" not in preds
    assert "flow" not in preds
    assert preds["rgb"].shape == (B, 3, C, H, W)
    print("test_halo_vlm_predict_visual_future_autoregressive passed.")


if __name__ == "__main__":
    print("Running tests...")
    test_dit_config_fallbacks()
    test_visual_dit_predictor_rgb_only_default()
    test_halo_vlm_multi_step_prediction_loss()
    test_halo_vlm_predict_visual_future_autoregressive()
    print("All tests passed successfully!")
