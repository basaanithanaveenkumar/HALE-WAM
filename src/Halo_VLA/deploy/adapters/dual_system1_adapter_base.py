"""Base of the System 1 adapters: inputs start from the System 2 latent."""

from __future__ import annotations

from typing import Any

from Halo_VLA.deploy.adapters.halo_adapter_base import HaloAdapterBase


class DualSystem1AdapterBase(HaloAdapterBase):
    """Shared System 1 geometry.

    Args:
        latent_tokens: latent length ``K`` from System 2 - an int (static: ``act_token`` = 1,
            ``multi_act`` = ``num_act_tokens``, ``pooled`` = 1) or ``[min, opt, max]`` for
            prompt-length latents (``last_hidden``, ``language``, ``middle_layer``). Default:
            static for query/pooled latents, ``[1, 16, 512]`` otherwise.
    """

    requires_dual = True

    def __init__(self, latent_tokens: int | list[int] | None = None, **base: Any) -> None:
        super().__init__(**base)
        self._latent_tokens = latent_tokens

    def latent_dim(self) -> int | list[int]:
        """Latent length ``K`` as a static int or a ``[min, opt, max]`` dynamic range.

        Uses ``latent_tokens`` when given, else the registered latent extractor's ``static_length``.
        """
        if self._latent_tokens is not None:
            return self._latent_tokens
        from dual_system.latent.latent_registry import (
            LATENT_EXTRACTORS,  # lazy: import roots are set up by Halo_VLA.deploy
        )

        static = LATENT_EXTRACTORS.get(self.dual_config().latent).static_length(self.dual_config())
        return static if static is not None else [1, 16, 512]

    def observation_tokens(self) -> int:
        """Number of observation tokens System 1's encoder emits (the resampler's count or the conv grid)."""
        dual = self.dual_config()
        if dual.obs_resampler_tokens:
            return int(dual.obs_resampler_tokens)
        grid = max(1, self.model_config().img_size // 16)
        return grid * grid

    def latent_inputs(self) -> list[dict[str, Any]]:
        """Input specs shared by the System 1 graphs: latent, latent mask, current image, state."""
        cfg, dual = self.model_config(), self.dual_config()
        return [
            {
                "name": "latent",
                "shape": ["batch", "latent_len", dual.system1_dim],
                "dtype": "float32",
                "value_range": [-1.0, 1.0],
            },
            {
                "name": "latent_mask",
                "shape": ["batch", "latent_len"],
                "dtype": "int64",
                "value_range": [1, 2],
            },
            {
                "name": "image",
                "shape": ["batch", 3, cfg.img_size, cfg.img_size],
                "dtype": "float32",
                "value_range": [0.0, 1.0],
            },
            {
                "name": "state",
                "shape": ["batch", cfg.state_dim],
                "dtype": "float32",
                "value_range": [-1.0, 1.0],
            },
        ]
