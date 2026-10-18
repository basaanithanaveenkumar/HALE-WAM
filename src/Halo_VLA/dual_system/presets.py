"""The published dual-system designs as registered presets.

Pattern: Registry. A preset is added by calling :func:`register_preset` with a
:class:`~dual_system.config.DualSystemConfig`; nothing else (CLI choices, resolution, docs) has
to change.

Example:
    >>> from dual_system.presets import DUAL_SYSTEM_PRESETS
    >>> DUAL_SYSTEM_PRESETS.get("lcb").policy
    'diffusion'
"""

from __future__ import annotations

from hale_core.registry import NamedRegistry

from dual_system.config import DualSystemConfig

#: ``preset name -> DualSystemConfig`` (iterates in registration order).
DUAL_SYSTEM_PRESETS = NamedRegistry("dual-system preset")


def register_preset(config: DualSystemConfig) -> DualSystemConfig:
    """Register ``config`` under ``config.name``.

    Args:
        config: the preset; its ``name`` is the registry key.

    Returns:
        ``config`` unchanged, so calls can be nested.

    Raises:
        hale_core.exceptions.RegistryError: if the name is already registered.
    """
    DUAL_SYSTEM_PRESETS.add(config.name, config)
    return config


# The designs of the OpenHelix survey (Table 1); ``none`` (no dual system) is not a preset.
register_preset(DualSystemConfig(
    name="basic", latent="act_token", system2_training="finetune", policy="flow_matching",
    notes="first-principles baseline: one <ACT> latent, flow-matching System 1"))
register_preset(DualSystemConfig(
    name="lcb", latent="act_token", system2_training="finetune", policy="diffusion", align_weight=1.0,
    async_steps=10, notes="LCB (Shentu et al., 2024, arXiv:2405.04798): <ACT> latent, LoRA System 2, "
    "diffusion actor, CLIP-style alignment, synchronous training / asynchronous testing"))
register_preset(DualSystemConfig(
    name="dp_vla", latent="last_hidden", system2_training="frozen", policy="regression",
    notes="DP-VLA (Han et al., CoRL 2024): frozen System 2, last-layer vision+language states, "
    "transformer policy trained from scratch"))
register_preset(DualSystemConfig(
    name="hirt", latent="pooled", system2_training="finetune", policy="regression",
    system2_frame_delay=1, async_steps=4,
    notes="HiRT (Zhang et al., 2024, arXiv:2410.05273): MAP-pooled last-layer latent, LoRA System 2, "
    "RT-1-style policy, delayed System 2 view during training"))
register_preset(DualSystemConfig(
    name="robodual", latent="multi_act", num_act_tokens=4, multi_act_with_language=True,
    system2_training="finetune", policy="diffusion", obs_resampler_tokens=16,
    notes="RoboDual (Bu et al., 2024, arXiv:2410.08001): action + task latents, DiT policy, "
    "Perceiver-resampled observations"))
register_preset(DualSystemConfig(
    name="dexvla", latent="language", system2_training="finetune", policy="diffusion",
    notes="DexVLA (Wen et al., 2025, arXiv:2502.05855): language latent, LoRA System 2, "
    "scaled diffusion policy"))
register_preset(DualSystemConfig(
    name="openhelix", latent="act_token", system2_training="prompt_tuning", policy="diffusion",
    projector="linear", aux_weight=1.0, prealign_steps=2000, async_steps=10,
    notes="OpenHelix (Cui et al., 2025, arXiv:2505.03912): frozen MLLM + prompt-tuned <ACT>, "
    "auxiliary action head, two-stage pre-alignment"))

#: Live ``name -> DualSystemConfig`` mapping (the registry's own table; kept for ``PRESETS[...]`` users).
PRESETS: dict[str, DualSystemConfig] = DUAL_SYSTEM_PRESETS.items

__all__ = ["DUAL_SYSTEM_PRESETS", "PRESETS", "register_preset"]
