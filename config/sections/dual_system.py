"""``model.dual_system``: slow System 2 / fast System 1 (docs/dual_system.md)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from config.sections.base import VLASection

PRESET_NAMES = (
    "none",
    "custom",
    "basic",
    "lcb",
    "dp_vla",
    "hirt",
    "robodual",
    "dexvla",
    "openhelix",
)


class DualSystemSection(VLASection):
    """``preset`` picks a published design (or ``none`` / ``custom``); every other field set here
    overrides that preset (``dual_system.config.DualSystemConfig``). Unset fields keep the
    preset's value.

    YAML example::

        model:
          dual_system:
            preset: openhelix
            async_steps: 60
            prealign_steps: 1000
    """

    preset: Literal[PRESET_NAMES] = "none"
    latent: (
        Literal["act_token", "multi_act", "last_hidden", "pooled", "language", "middle_layer"]
        | None
    ) = None
    num_act_tokens: int | None = None
    multi_act_with_language: bool | None = None
    middle_layer: int | None = None
    system2_training: Literal["frozen", "prompt_tuning", "finetune"] | None = None
    policy: Literal["regression", "diffusion", "flow_matching"] | None = None
    system1_dim: int | None = None
    system1_layers: int | None = None
    system1_heads: int | None = None
    obs_resampler_tokens: int | None = None
    projector: Literal["linear", "mlp"] | None = None
    align_weight: float | None = None
    aux_weight: float | None = None
    prealign_steps: int | None = None
    async_steps: int | None = None
    system2_frame_delay: int | None = None
    lm_weight: float | None = None
    diffusion_train_steps: int | None = None
    sample_steps: int | None = None
    system1_checkpoint: str | None = None

    def options(self) -> dict[str, Any]:
        """Preset overrides given in the YAML (``null`` is kept: it can mean "no resampler")."""
        return {k: v for k, v in self.explicitly_set().items() if k != "preset"}

    @model_validator(mode="after")
    def _resolves(self) -> DualSystemSection:
        from dual_system.resolution import resolve_dual_system_config

        resolve_dual_system_config(
            self.preset, self.options() or None
        )  # raises on bad combinations
        return self

    def train_arguments(self) -> dict[str, Any]:
        """``--dual_system`` / ``--dual_system_option`` values for the keys the YAML sets."""
        out: dict[str, Any] = {}
        if "preset" in self.model_fields_set:
            out["dual_system"] = self.preset
        options = self.options()
        if options:
            out["dual_system_option"] = [
                f"{k}={'none' if v is None else v}" for k, v in options.items()
            ]
        return out
