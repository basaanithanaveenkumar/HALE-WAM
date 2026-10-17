"""VisionEncoderInfo: what the rest of the VLA needs to know about the encoder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionEncoderInfo:
    """What the rest of the VLA needs to know about the encoder."""

    output_dim: int
    num_tokens: int
