"""DiTConfig: diT video / depth / flow predictor."""

from __future__ import annotations

from config.sections.base import VLASection


class DiTConfig(VLASection):
    """DiT video / depth / flow predictor."""

    OVERRIDES = {"patch_size": "dit_patch_size", "hidden_size": "dit_hidden_size", "depth": "dit_depth",
                 "num_heads": "dit_num_heads", "mlp_ratio": "dit_mlp_ratio", "num_sample_steps": "dit_num_sample_steps"}

    patch_size: int = 8
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_sample_steps: int = 75
