"""Hale-VLM parity for Halo-VLA: configurable vision encoders, validation evaluator, data split."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from config import HaloVLMConfig  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402
from training import HaloVLATrainingStep, VLAEvaluator, split_validation  # noqa: E402

from tests.test_llm_backbone import batch_for  # noqa: E402


def small(**kw) -> HaloVLMConfig:
    base = dict(img_size=32, patch_size=16, emb_dim=64, vit_num_layers=1, vit_num_heads=4, vit_mlp_dim=64,
                dec_num_layers=1, dec_num_heads=4, dec_mlp_dim=64, use_moe=False, action_dim=4, state_dim=3,
                action_chunk_size=2, enable_visual_dit=False, flow_hidden_dim=32, flow_num_ode_steps=2)
    base.update(kw)
    return HaloVLMConfig(**base)


# ---------------------------------------------------------------------------- vision encoders
def test_builtin_vit_is_the_default_and_trains() -> None:
    cfg = small()
    model = HaloVLM(cfg)
    assert cfg.vision_num_tokens == 4 and cfg.num_image_tokens_per_image == 4
    assert all(p.requires_grad for p in model.vis_enc.parameters())


def test_local_hale_backbone_sets_width_tokens_and_stays_frozen() -> None:
    pytest.importorskip("hale_core")
    cfg = small(vision_encoder="local", vision_backbone="vit_tiny")
    model = HaloVLM(cfg)
    width = model.vis_enc.output_dim
    assert cfg.proj_vision_dim == width and cfg.vision_num_tokens == 4
    assert not any(p.requires_grad for p in model.vis_enc.backbone.parameters())  # frozen by default
    model.train()
    assert not model.vis_enc.backbone.training  # frozen backbone stays in eval mode
    out = HaloVLATrainingStep(model)(batch_for(cfg))  # language loss uses the measured token count
    assert torch.isfinite(out.total)


def test_local_backbone_can_be_fine_tuned() -> None:
    pytest.importorskip("hale_core")
    cfg = small(vision_encoder="local", vision_backbone="vit_tiny", vision_freeze=False)
    model = HaloVLM(cfg)
    assert any(p.requires_grad for p in model.vis_enc.backbone.parameters())


def test_siglip_encoder_renormalises_and_checks_resolution() -> None:
    transformers = pytest.importorskip("transformers")
    tiny = transformers.SiglipVisionConfig(image_size=32, patch_size=16, hidden_size=32, intermediate_size=64,
                                           num_hidden_layers=1, num_attention_heads=4)
    with patch.object(transformers.SiglipVisionModel, "from_pretrained",
                      side_effect=lambda *a, **k: transformers.SiglipVisionModel(tiny)), \
         patch.object(transformers.AutoImageProcessor, "from_pretrained", side_effect=OSError):
        cfg = small(vision_encoder="siglip")
        model = HaloVLM(cfg)
        enc = model.vis_enc
        assert cfg.vision_num_tokens == 4 and cfg.proj_vision_dim == 32 and enc.frozen
        raw = torch.rand(1, 3, 32, 32)  # pixels in [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        dataset_norm = (raw - mean) / std
        assert torch.allclose(dataset_norm * enc.scale + enc.shift, (raw - 0.5) / 0.5, atol=1e-5)
        with pytest.raises(ValueError, match="expects 32px"):
            HaloVLM(small(vision_encoder="siglip", img_size=64))


# ---------------------------------------------------------------------------- evaluation
def test_evaluator_reports_losses_and_action_metrics() -> None:
    torch.manual_seed(0)
    cfg = small()
    model = HaloVLM(cfg).train()
    result = VLAEvaluator(cfg.action_dim).run(HaloVLATrainingStep(model), [batch_for(cfg), batch_for(cfg)],
                                              torch.device("cpu"))
    for key in ("val/loss_total", "val/loss_act", "val/action_l1", "val/action_mse", "val/chunk_acc_0.10",
                "val/pred_smoothness"):
        assert key in result and torch.isfinite(torch.tensor(result[key])), key
    assert not any("_dim_" in k for k in result) and model.training  # train mode restored


def test_split_validation_is_deterministic_and_disjoint() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.arange(20)), batch_size=4)
    train_a, val_a = split_validation(loader, 0.25, seed=3)
    train_b, val_b = split_validation(loader, 0.25, seed=3)
    assert len(val_a.dataset) == 5 and len(train_a.dataset) == 15
    assert list(val_a.dataset.indices) == list(val_b.dataset.indices)
    assert not set(val_a.dataset.indices) & set(train_a.dataset.indices)
    with pytest.raises(ValueError):
        split_validation(loader, 1.0)


def test_trajectory_smoothness_uses_true_second_differences() -> None:
    from metrics import TrajectorySmoothnessMetric

    t = torch.arange(6, dtype=torch.float32)
    traj = (t**2).view(1, 6, 1).expand(2, 6, 3)  # second difference of t^2 is exactly 2
    metric = TrajectorySmoothnessMetric()
    metric.update(pred=traj, target=traj)
    out = metric.compute()
    assert out["pred_smoothness"] == pytest.approx(4.0) and out["target_smoothness"] == pytest.approx(4.0)
