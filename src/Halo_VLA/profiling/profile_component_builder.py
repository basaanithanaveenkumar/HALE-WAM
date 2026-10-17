"""Builds one stub component and its dummy input for ``scripts/profile_model.py``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from hale_core.registry import NamedRegistry

from profiling.stubs import StubActionDecoder, StubDecoderTransformer, StubHaloVLA, StubViT

#: ``--component`` name -> builder; add a component by registering a function here.
PROFILE_COMPONENTS = NamedRegistry("profile component")

#: input resolution of the stub ViT (matches the real 224-px Halo-VLA encoder)
IMAGE_SIZE = 224
#: 14x14 ViT patches plus a short text/action tail, the decoder's typical sequence length
DECODER_SEQ_LEN = 196 + 8


@dataclass(frozen=True)
class ProfileComponentBuilder:
    """Model width settings shared by every profiled component.

    Attributes:
        batch_size: dummy batch size.
        emb_dim: model width (1024 for full scale).
        n_layers: transformer depth (6 for full scale).
        n_heads: attention heads (8 for full scale).
        mlp_dim: MLP hidden width (2048 for full scale).
    """

    batch_size: int
    emb_dim: int
    n_layers: int
    n_heads: int
    mlp_dim: int

    def build(self, component: str) -> tuple[nn.Module, torch.Tensor | tuple[torch.Tensor, ...]]:
        """Return ``(model, dummy_input)`` for the registered ``component``.

        Raises:
            hale_core.exceptions.RegistryError: for an unknown component name.
        """
        return PROFILE_COMPONENTS.get(component)(self)


@PROFILE_COMPONENTS.register_module(name="vit")
def _build_vit(b: ProfileComponentBuilder):
    model = StubViT(emb_dim=b.emb_dim, n_layers=b.n_layers, n_heads=b.n_heads, mlp_dim=b.mlp_dim)
    return model, torch.randn(b.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)


@PROFILE_COMPONENTS.register_module(name="action")
def _build_action(b: ProfileComponentBuilder):
    return StubActionDecoder(emb_dim=b.emb_dim), torch.randn(b.batch_size, b.emb_dim)


@PROFILE_COMPONENTS.register_module(name="decoder")
def _build_decoder(b: ProfileComponentBuilder):
    model = StubDecoderTransformer(
        emb_dim=b.emb_dim, n_layers=b.n_layers, n_heads=b.n_heads, mlp_dim=b.mlp_dim
    )
    return model, torch.randn(b.batch_size, DECODER_SEQ_LEN, b.emb_dim)


@PROFILE_COMPONENTS.register_module(name="vla")
def _build_vla(b: ProfileComponentBuilder):
    model = StubHaloVLA(
        emb_dim=b.emb_dim, n_vit_layers=b.n_layers, n_dec_layers=b.n_layers,
        n_heads=b.n_heads, mlp_dim=b.mlp_dim,
    )
    images = torch.randn(b.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    return model, (images, torch.randn(b.batch_size, b.emb_dim))
