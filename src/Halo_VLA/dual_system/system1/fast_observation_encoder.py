"""FastObservationEncoder: a light convolutional encoder for the *current* camera frame (runs at control rate)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FastObservationEncoder(nn.Module):
    """A light convolutional encoder for the *current* camera frame (runs at control rate).

    Three strided convolutions reduce ``(B, 3, H, W)`` to an ``H/16 x W/16`` grid of ``dim``-wide
    tokens with learned 2D positions - cheap enough to run every control step, unlike System 2.
    """

    def __init__(self, dim: int, img_size: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 64, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(64, dim, 2, stride=2),
        )
        grid = max(1, img_size // 16)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode the current frame into tokens.

        Args:
            image: ``(B, 3, H, W)``.

        Returns:
            ``(B, N, dim)`` tokens with learned positions.
        """
        tokens = self.stem(image).flatten(2).transpose(1, 2)  # (B, N, dim)
        return tokens + self.pos[:, : tokens.size(1)]
