"""Architecture-equivalent stub components for profiling Halo-VLA without real weights."""

from .stub_action_decoder import StubActionDecoder  # noqa: F401
from .stub_decoder_transformer import StubDecoderTransformer  # noqa: F401
from .stub_halo_vla import StubHaloVLA  # noqa: F401
from .stub_patch_embed import StubPatchEmbed  # noqa: F401
from .stub_transformer_block import StubTransformerBlock  # noqa: F401
from .stub_vit import StubViT  # noqa: F401

__all__ = [
    "StubActionDecoder",
    "StubDecoderTransformer",
    "StubHaloVLA",
    "StubPatchEmbed",
    "StubTransformerBlock",
    "StubViT",
]
