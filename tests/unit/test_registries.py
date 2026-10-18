"""Registries replace if/elif chains: names in configs resolve to registered implementations."""

from __future__ import annotations

import argparse

import pytest
import torch
from exceptions import HaloVLAConfigError
from hale_core.exceptions import RegistryError

from config import HaloVLMConfig
from dual_system import (
    DUAL_SYSTEM_PRESETS,
    LATENT_EXTRACTORS,
    PRESETS,
    SYSTEM1_POLICIES,
    DualSystemConfig,
    register_preset,
    resolve_dual_system_config,
)
from dual_system.config import LATENT_TYPES, POLICY_TYPES
from dual_system.latent import BRIDGE_PROJECTORS
from models.image_proj import IMAGE_PROJECTORS, PROJECTOR_TYPES, build_image_projector
from models.vision_encoders import ENCODER_TYPES, VISION_ENCODERS, build_vision_encoder
from training.train_datasets import TRAIN_DATASETS, build_train_loader


def test_config_tuples_match_the_registries() -> None:
    assert set(LATENT_TYPES) == set(LATENT_EXTRACTORS.keys())
    assert set(POLICY_TYPES) == set(SYSTEM1_POLICIES.keys())
    assert set(BRIDGE_PROJECTORS.keys()) == {"linear", "mlp"}
    assert set(PROJECTOR_TYPES) == set(IMAGE_PROJECTORS.keys()) == {"mlp", "qformer", "gated_cross_attention"}
    assert set(ENCODER_TYPES) == set(VISION_ENCODERS.keys()) == {"vit", "siglip", "clip", "local"}
    assert set(TRAIN_DATASETS.keys()) == {"eo", "droid", "moma"}


def test_presets_are_the_registry_mapping() -> None:
    assert PRESETS is DUAL_SYSTEM_PRESETS.items
    assert set(PRESETS) == {"basic", "lcb", "dp_vla", "hirt", "robodual", "dexvla", "openhelix"}
    assert all(cfg.name == name for name, cfg in PRESETS.items())


def test_a_registered_preset_resolves_by_name_and_duplicates_raise() -> None:
    preset = DualSystemConfig(name="unit_test_preset", latent="pooled", policy="regression")
    try:
        assert register_preset(preset) is preset
        assert resolve_dual_system_config("unit_test_preset").latent == "pooled"
        with pytest.raises(RegistryError):
            register_preset(preset)
    finally:
        DUAL_SYSTEM_PRESETS.items.pop("unit_test_preset", None)


def test_unknown_names_raise_actionable_config_errors() -> None:
    with pytest.raises(HaloVLAConfigError, match="unknown dual_system 'helix'.*openhelix"):
        resolve_dual_system_config("helix")
    with pytest.raises(HaloVLAConfigError, match="Unknown proj_type 'perceiver'"):
        build_image_projector(_config_with(proj_type="perceiver"))
    with pytest.raises(HaloVLAConfigError, match="vision_encoder must be one of"):
        build_vision_encoder(_config_with(vision_encoder="resnet-from-nowhere"))
    with pytest.raises(HaloVLAConfigError, match="unknown --dataset 'imagenet'"):
        build_train_loader(argparse.Namespace(dataset="imagenet"), HaloVLMConfig(), None)


def _config_with(**fields) -> HaloVLMConfig:
    """A valid config whose fields are overwritten afterwards (``__post_init__`` would reject them)."""
    config = HaloVLMConfig()
    for name, value in fields.items():
        setattr(config, name, value)
    return config


@pytest.mark.parametrize("proj_type", sorted(PROJECTOR_TYPES))
def test_every_projector_builds_and_maps_vision_tokens(proj_type: str) -> None:
    config = HaloVLMConfig(img_size=32, patch_size=16, emb_dim=32, proj_type=proj_type, qformer_hidden_dim=32,
                           qformer_num_heads=4, gated_xattn_hidden_dim=32, gated_xattn_num_heads=4)
    out = build_image_projector(config)(torch.randn(2, 4, 32))
    assert out.shape[0] == 2 and out.shape[-1] == 32


def test_the_vit_encoder_reports_its_token_geometry_without_a_forward_pass() -> None:
    config = HaloVLMConfig(img_size=32, patch_size=16, emb_dim=32, vit_num_layers=1, vit_num_heads=4, vit_mlp_dim=32)
    encoder, info = build_vision_encoder(config)
    assert (info.output_dim, info.num_tokens) == (32, 4)
    assert encoder(torch.zeros(1, 3, 32, 32)).shape == (1, 4, 32)


def test_train_dataset_builders_require_their_data_roots() -> None:
    args = argparse.Namespace(dataset="droid", droid_data_root=None)
    with pytest.raises(HaloVLAConfigError, match="--droid_data_root"):
        build_train_loader(args, HaloVLMConfig(), None)
    args = argparse.Namespace(dataset="moma", moma_data_root="")
    with pytest.raises(HaloVLAConfigError, match="--moma_data_root"):
        build_train_loader(args, HaloVLMConfig(), None)
