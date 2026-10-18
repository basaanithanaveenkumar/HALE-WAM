"""Latent extractors: what System 2 hands to System 1, one registered class per ``latent`` value."""

from __future__ import annotations

import pytest
import torch

from dual_system import LATENT_EXTRACTORS, DualSystemConfig, build_latent_extractor
from dual_system.latent import ActTokenLatent, MiddleLayerLatent, PooledLatent
from exceptions import ModelStateError

B, PREFIX, TEXT, EXTRA, D = 2, 3, 4, 2, 16


def encoding(layer: bool = False) -> dict:
    """A System 2 encoding: ``PREFIX`` image tokens, ``TEXT`` instruction tokens, ``EXTRA`` <ACT> queries."""
    total = PREFIX + TEXT + EXTRA
    gen = torch.Generator().manual_seed(0)
    return {
        "hidden": torch.randn(B, total, D, generator=gen),
        "mask": torch.ones(B, total, dtype=torch.long),
        "num_prefix": PREFIX,
        "text_len": TEXT,
        "num_extra": EXTRA,
        "layer_hidden": torch.randn(B, total, 2 * D, generator=gen) if layer else None,
    }


def config(**kw) -> DualSystemConfig:
    return DualSystemConfig(**kw)


@pytest.mark.parametrize(
    "latent, expected_tokens",
    [
        ("act_token", 1),
        ("multi_act", EXTRA),
        ("last_hidden", PREFIX + TEXT + EXTRA),
        ("language", TEXT),
        ("pooled", 1),
    ],
)
def test_each_extractor_returns_tokens_and_a_matching_mask(
    latent: str, expected_tokens: int
) -> None:
    cfg = config(latent=latent, num_act_tokens=EXTRA)
    extractor = build_latent_extractor(cfg, D, 2 * D)
    enc = encoding()
    if latent == "act_token":
        enc["num_extra"] = 1
        enc["hidden"], enc["mask"] = enc["hidden"][:, :-1], enc["mask"][:, :-1]
    tokens, mask = extractor(enc)
    assert tokens.shape == (B, expected_tokens, D) and mask.shape == (B, expected_tokens)


def test_multi_act_with_language_appends_the_instruction_states() -> None:
    extractor = build_latent_extractor(
        config(latent="multi_act", num_act_tokens=EXTRA, multi_act_with_language=True), D, 2 * D
    )
    tokens, mask = extractor(encoding())
    assert (
        isinstance(extractor, ActTokenLatent)
        and tokens.shape == (B, EXTRA + TEXT, D)
        and mask.shape[1] == EXTRA + TEXT
    )


def test_middle_layer_uses_the_raw_decoder_width_and_needs_a_captured_layer() -> None:
    extractor = build_latent_extractor(config(latent="middle_layer"), D, 2 * D)
    assert (
        isinstance(extractor, MiddleLayerLatent)
        and extractor.width == 2 * D
        and extractor.needs_layer_capture
    )
    assert extractor(encoding(layer=True))[0].shape[-1] == 2 * D
    with pytest.raises(ModelStateError, match="capture a layer"):
        extractor(encoding(layer=False))


def test_pooled_latent_attends_over_all_states_with_the_padding_masked() -> None:
    extractor = PooledLatent(D, heads=4)
    enc = encoding()
    enc["mask"][:, -2:] = 0
    tokens, mask = extractor(enc)
    assert tokens.shape == (B, 1, D) and mask.tolist() == [[1], [1]]


@pytest.mark.parametrize(
    "latent, kwargs, expected",
    [
        ("act_token", {}, 1),
        ("multi_act", {"num_act_tokens": 4}, 4),
        ("multi_act", {"num_act_tokens": 4, "multi_act_with_language": True}, None),
        ("pooled", {}, 1),
        ("last_hidden", {}, None),
        ("language", {}, None),
        ("middle_layer", {}, None),
    ],
)
def test_static_length_tells_deployment_whether_the_latent_length_is_fixed(
    latent, kwargs, expected
) -> None:
    cfg = config(latent=latent, **kwargs)
    assert LATENT_EXTRACTORS.get(latent).static_length(cfg) == expected
