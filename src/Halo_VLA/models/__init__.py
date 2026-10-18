"""
Halo-VLA: Vision-Language Assistant Model
A PyTorch implementation of a Vision-Language Model (VLA) architecture.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from .action_decoder import ActionDecoder
from .DiT import DiT, DiTBlock, ScalarTimestepEmbedder
from .dit_frame_prediction import VisualDiTPredictor, make_dit_config
from .halo_vla import HaloVLM
from .lm_head import LMHead
from .state_encoder import StateEncoder
from .transformer import DecoderTransformer
from .vit import VisTransformer

__all__ = [
    "HaloVLM",
    "DecoderTransformer",
    "VisTransformer",
    "LMHead",
    "ActionDecoder",
    "StateEncoder",
    "DiT",
    "DiTBlock",
    "ScalarTimestepEmbedder",
    "VisualDiTPredictor",
    "make_dit_config",
]
