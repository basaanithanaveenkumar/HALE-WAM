"""build_model_config maps ``scripts/train.py`` arguments onto a validated HaloVLMConfig."""

from __future__ import annotations

import pytest
from exceptions import HaloVLAConfigError

from training import build_model_config
from training.train_arg_parser import parse_args


def test_defaults_give_the_dataclass_defaults() -> None:
    from config import HaloVLMConfig

    config = build_model_config(parse_args([]))
    default = HaloVLMConfig()
    assert (config.emb_dim, config.proj_type, config.vision_encoder) == (default.emb_dim, default.proj_type,
                                                                          default.vision_encoder)
    assert config.dual_system == "none" and config.dual_system_options is None


def test_flags_reach_the_config() -> None:
    args = parse_args(["--action_dim", "7", "--state_dim", "8", "--projector_type", "qformer",
                       "--num_query_tokens", "12", "--visual_loss_weight", "2.5", "--vision_encoder", "local",
                       "--vision_backbone", "vit_tiny", "--dual_system", "lcb",
                       "--dual_system_option", "async_steps=4"])
    config = build_model_config(args)
    assert (config.action_dim, config.state_dim, config.proj_type) == (7, 8, "qformer")
    assert config.qformer_num_queries == config.gated_xattn_num_latents == 12
    assert config.visual_loss_weight == 2.5 and config.vision_backbone == "vit_tiny"
    assert config.dual_system == "lcb" and config.dual_system_options == {"async_steps": 4}


def test_model_options_override_and_are_validated() -> None:
    assert build_model_config(parse_args(["--model_option", "emb_dim=128"])).emb_dim == 128
    with pytest.raises(HaloVLAConfigError, match="not_a_field"):
        build_model_config(parse_args(["--model_option", "not_a_field=1"]))
