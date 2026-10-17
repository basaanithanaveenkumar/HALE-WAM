"""Profiling helpers: architecture-equivalent stubs and the component builder registry."""

from profiling.profile_component_builder import PROFILE_COMPONENTS, ProfileComponentBuilder
from profiling.stubs import (
    StubActionDecoder,
    StubDecoderTransformer,
    StubHaloVLA,
    StubPatchEmbed,
    StubTransformerBlock,
    StubViT,
)

__all__ = [
    "PROFILE_COMPONENTS",
    "ProfileComponentBuilder",
    "StubActionDecoder",
    "StubDecoderTransformer",
    "StubHaloVLA",
    "StubPatchEmbed",
    "StubTransformerBlock",
    "StubViT",
]
