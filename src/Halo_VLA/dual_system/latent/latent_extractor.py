"""LatentExtractor: the abstract contract of every System 2 -> System 1 latent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from dual_system.config import DualSystemConfig


class LatentExtractor(nn.Module, ABC):
    """Maps System 2's encoding to latent ``tokens (B, K, D)`` and ``mask (B, K)``.

    Pattern: Strategy + Registry. Each subclass registers under the ``DualSystemConfig.latent``
    value(s) it implements in :data:`~dual_system.latent.latent_registry.LATENT_EXTRACTORS`;
    :func:`~dual_system.latent.common.build_latent_extractor` looks it up by name.

    Attributes:
        needs_layer_capture: whether System 2 must capture a middle decoder layer for this latent.

    Args:
        width: token width ``D`` of the latent.
    """

    needs_layer_capture: bool = False

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    @classmethod
    def from_config(
        cls, config: DualSystemConfig, emb_dim: int, layer_width: int
    ) -> LatentExtractor:
        """Build the extractor for ``config``.

        Args:
            config: dual-system settings.
            emb_dim: System 2's embedding width.
            layer_width: the decoder's raw hidden width (used by ``middle_layer``).

        Returns:
            The configured extractor (default: ``cls(emb_dim)``).
        """
        return cls(emb_dim)

    @classmethod
    def static_length(cls, config: DualSystemConfig) -> int | None:
        """Number of latent tokens when it does not depend on the prompt, else ``None``."""
        return None

    @abstractmethod
    def forward(self, enc: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the latent.

        Args:
            enc: System 2's encoding (``hidden``, ``mask``, ``num_prefix``, ``text_len``,
                ``num_extra``, ``layer_hidden``).

        Returns:
            ``(tokens (B, K, D), mask (B, K))`` with ``mask`` 1 for real tokens.
        """
