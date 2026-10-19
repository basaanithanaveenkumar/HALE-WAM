"""Name-selected components resolve through hale_core registries that match their config/CLI."""

from __future__ import annotations

import typing

import pytest

from dual_system.config import DualSystemConfig
from dual_system.latent import LATENT_EXTRACTORS, build_latent_extractor
from exceptions import HaloVLAConfigError
from models.image_proj import IMAGE_PROJECTORS, PROJECTOR_TYPES
from models.vision_encoders import ENCODER_TYPES, VISION_ENCODERS
from training.train_arg_parser import parse_args
from training.train_dataloaders import TRAIN_DATALOADERS, build_train_dataloader


def test_projector_registry_matches_the_config_choices() -> None:
    assert PROJECTOR_TYPES == ("mlp", "qformer", "gated_cross_attention")
    assert set(PROJECTOR_TYPES) == set(IMAGE_PROJECTORS)


def test_vision_encoder_registry_lists_every_encoder() -> None:
    assert set(ENCODER_TYPES) == set(VISION_ENCODERS) == {"vit", "siglip", "clip", "local"}


def test_latent_registry_covers_every_latent_type() -> None:
    hints = typing.get_type_hints(DualSystemConfig)
    assert set(typing.get_args(hints["latent"])) == set(LATENT_EXTRACTORS)


@pytest.mark.parametrize("latent", ["act_token", "multi_act", "last_hidden", "language", "pooled"])
def test_every_latent_builds(latent: str) -> None:
    extractor = build_latent_extractor(DualSystemConfig(latent=latent), emb_dim=16, layer_width=24)
    assert extractor.width == 16


def test_dataset_choices_match_the_dataloader_registry() -> None:
    import argparse

    from training import train_arg_parser

    parser_choices = None
    original = argparse.ArgumentParser.parse_args
    try:
        def capture(self, *a, **k):
            nonlocal parser_choices
            parser_choices = next(x.choices for x in self._actions if "--dataset" in x.option_strings)
            return original(self, *a, **k)

        argparse.ArgumentParser.parse_args = capture
        train_arg_parser.parse_args([])
    finally:
        argparse.ArgumentParser.parse_args = original
    assert set(parser_choices) == set(TRAIN_DATALOADERS)


def test_local_datasets_require_a_data_root() -> None:
    args = parse_args(["--dataset", "droid"])
    with pytest.raises(HaloVLAConfigError, match="droid_data_root"):
        build_train_dataloader(args, img_size=32)
