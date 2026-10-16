"""Dual-system VLA configuration: one dataclass, one preset per published design.

A dual-system VLA (OpenHelix survey, Cui et al., 2025, arXiv:2505.03912) is defined by the answers
to a few design questions. Each field below is one of them; each preset is one paper's answers.

====================  =================================================================
design question       field(s)
====================  =================================================================
latent from System 2  ``latent``: act_token | multi_act | last_hidden | pooled |
                      language | middle_layer  (+ ``num_act_tokens``, ``middle_layer``)
System 2 training     ``system2_training``: frozen | prompt_tuning | finetune
System 1 policy       ``policy``: regression | diffusion | flow_matching
System 1 sensing      ``obs_resampler_tokens`` (Perceiver resampler), proprioception always
integration           ``projector``: linear | mlp, ``align_weight`` (LCB's CLIP-style loss),
                      ``aux_weight`` (OpenHelix auxiliary action head),
                      ``prealign_steps`` (OpenHelix two-stage training)
asynchrony            ``async_steps`` (System 1 steps per System 2 step at inference),
                      ``system2_frame_delay`` (stale System 2 view during training, HiRT)
====================  =================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal
from exceptions import HaloVLAConfigError

LatentType = Literal["act_token", "multi_act", "last_hidden", "pooled", "language", "middle_layer"]
System2Training = Literal["frozen", "prompt_tuning", "finetune"]
PolicyType = Literal["regression", "diffusion", "flow_matching"]

LATENT_TYPES = ("act_token", "multi_act", "last_hidden", "pooled", "language", "middle_layer")
SYSTEM2_TRAINING = ("frozen", "prompt_tuning", "finetune")
POLICY_TYPES = ("regression", "diffusion", "flow_matching")


@dataclass
class DualSystemConfig:
    """Everything that distinguishes one dual-system design from another.

    Attributes:
        name: preset this config came from (``custom`` when built by hand).
        latent: what System 2 hands to System 1 (see module docstring).
        num_act_tokens: learnable ``<ACT>`` queries appended to System 2's input
            (``act_token`` uses 1; ``multi_act`` uses this many).
        multi_act_with_language: ``multi_act`` also passes the instruction's hidden states
            (RoboDual's task latent + action latent).
        middle_layer: decoder layer for ``latent: middle_layer`` (``None`` = middle of the stack).
        system2_training: ``frozen`` (no System 2 gradients), ``prompt_tuning`` (only the
            ``<ACT>`` queries train; OpenHelix), ``finetune`` (System 2 trains as configured -
            with a pretrained LLM backbone that is ``llm_finetune``, e.g. LoRA).
        policy: System 1 action head: ``regression`` (direct chunk prediction, DP-VLA / HiRT),
            ``diffusion`` (DDPM noise prediction, LCB / RoboDual / DexVLA / OpenHelix) or
            ``flow_matching`` (rectified flow, pi0-style).
        system1_dim: System 1 width.
        system1_layers: transformer layers of the action head.
        system1_heads: attention heads of System 1.
        obs_resampler_tokens: compress System 1's image tokens to this many with a Perceiver
            resampler (RoboDual); ``None`` keeps every patch token.
        projector: latent projector ``linear`` (OpenHelix) or 2-layer ``mlp``.
        align_weight: weight of the latent-to-instruction alignment loss (LCB's CLIP loss).
        aux_weight: weight of the auxiliary action head on the latent (OpenHelix AUX).
        prealign_steps: optimiser steps with System 1 frozen while the projector / prompt align
            (OpenHelix stage 1); ``0`` = single-stage training.
        async_steps: System 1 control steps per System 2 update at inference.
        system2_frame_delay: during training System 2 sees frames this many steps older than
            System 1 (HiRT-style latency simulation); ``0`` = synchronous training.
        lm_weight: weight of System 2's language-modelling loss (keeps it a language model).
        diffusion_train_steps: DDPM noise levels.
        sample_steps: denoising / ODE steps when sampling actions.
        system1_checkpoint: load pretrained System 1 weights (``Policy(P)`` in OpenHelix).
    """

    name: str = "custom"
    latent: LatentType = "act_token"
    num_act_tokens: int = 1
    multi_act_with_language: bool = False
    middle_layer: int | None = None
    system2_training: System2Training = "finetune"
    policy: PolicyType = "flow_matching"
    system1_dim: int = 256
    system1_layers: int = 4
    system1_heads: int = 8
    obs_resampler_tokens: int | None = None
    projector: Literal["linear", "mlp"] = "mlp"
    align_weight: float = 0.0
    aux_weight: float = 0.0
    prealign_steps: int = 0
    async_steps: int = 1
    system2_frame_delay: int = 0
    lm_weight: float = 0.0
    diffusion_train_steps: int = 100
    sample_steps: int = 10
    system1_checkpoint: str | None = None
    notes: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        checks = (("latent", LATENT_TYPES), ("system2_training", SYSTEM2_TRAINING), ("policy", POLICY_TYPES))
        for attr, allowed in checks:
            if getattr(self, attr) not in allowed:
                raise HaloVLAConfigError(f"dual_system.{attr} must be one of {allowed}, got {getattr(self, attr)!r}")
        if self.projector not in ("linear", "mlp"):
            raise HaloVLAConfigError(f"dual_system.projector must be linear or mlp, got {self.projector!r}")
        if self.num_act_tokens < 1 or self.async_steps < 1 or self.system2_frame_delay < 0:
            raise HaloVLAConfigError("num_act_tokens and async_steps must be >= 1, system2_frame_delay >= 0")
        if self.latent == "act_token":
            self.num_act_tokens = 1
        if self.system2_training == "prompt_tuning" and self.latent not in ("act_token", "multi_act"):
            raise HaloVLAConfigError("prompt_tuning trains the <ACT> queries: use latent act_token or multi_act")
        if self.align_weight > 0 and self.latent == "middle_layer":
            raise HaloVLAConfigError("align_weight needs a latent in the model's embedding space (not middle_layer)")

    @property
    def uses_act_queries(self) -> bool:
        return self.latent in ("act_token", "multi_act")


#: The published designs (OpenHelix survey, Table 1) as presets; ``none`` disables the dual system.
PRESETS: dict[str, DualSystemConfig] = {
    "basic": DualSystemConfig(
        name="basic", latent="act_token", system2_training="finetune", policy="flow_matching",
        notes="first-principles baseline: one <ACT> latent, flow-matching System 1"),
    "lcb": DualSystemConfig(
        name="lcb", latent="act_token", system2_training="finetune", policy="diffusion", align_weight=1.0,
        async_steps=10, notes="LCB (Shentu et al., 2024, arXiv:2405.04798): <ACT> latent, LoRA System 2, "
        "diffusion actor, CLIP-style alignment, synchronous training / asynchronous testing"),
    "dp_vla": DualSystemConfig(
        name="dp_vla", latent="last_hidden", system2_training="frozen", policy="regression",
        notes="DP-VLA (Han et al., CoRL 2024): frozen System 2, last-layer vision+language states, "
        "transformer policy trained from scratch"),
    "hirt": DualSystemConfig(
        name="hirt", latent="pooled", system2_training="finetune", policy="regression",
        system2_frame_delay=1, async_steps=4,
        notes="HiRT (Zhang et al., 2024, arXiv:2410.05273): MAP-pooled last-layer latent, LoRA System 2, "
        "RT-1-style policy, delayed System 2 view during training"),
    "robodual": DualSystemConfig(
        name="robodual", latent="multi_act", num_act_tokens=4, multi_act_with_language=True,
        system2_training="finetune", policy="diffusion", obs_resampler_tokens=16,
        notes="RoboDual (Bu et al., 2024, arXiv:2410.08001): action + task latents, DiT policy, "
        "Perceiver-resampled observations"),
    "dexvla": DualSystemConfig(
        name="dexvla", latent="language", system2_training="finetune", policy="diffusion",
        notes="DexVLA (Wen et al., 2025, arXiv:2502.05855): language latent, LoRA System 2, "
        "scaled diffusion policy"),
    "openhelix": DualSystemConfig(
        name="openhelix", latent="act_token", system2_training="prompt_tuning", policy="diffusion",
        projector="linear", aux_weight=1.0, prealign_steps=2000, async_steps=10,
        notes="OpenHelix (Cui et al., 2025, arXiv:2505.03912): frozen MLLM + prompt-tuned <ACT>, "
        "auxiliary action head, two-stage pre-alignment"),
}
NONE = "none"


def parse_overrides(items: list[str] | None) -> dict:
    """``["latent=pooled", "async_steps=4"]`` -> typed overrides for :class:`DualSystemConfig`."""
    types = {f.name: f.type for f in fields(DualSystemConfig)}
    out: dict = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or key not in types:
            raise HaloVLAConfigError(f"bad dual-system option {item!r}; use KEY=VALUE with KEY in {sorted(types)}")
        kind = str(types[key])
        if raw.lower() in ("none", "null"):
            value = None
        elif "bool" in kind:
            value = raw.lower() in ("1", "true", "yes")
        elif "int" in kind:
            value = int(raw)
        elif "float" in kind:
            value = float(raw)
        else:
            value = raw
        out[key] = value
    return out


def resolve_dual_system_config(name: str | None, overrides: dict | None = None) -> DualSystemConfig | None:
    """The preset ``name`` (or ``None`` for ``none``) with ``overrides`` applied.

    ``name="custom"`` starts from the dataclass defaults, so any combination can be built from
    overrides alone.
    """
    if name is None or name == NONE:
        if overrides:
            raise HaloVLAConfigError("dual_system options given but dual_system is 'none'")
        return None
    if name == "custom":
        base = DualSystemConfig()
    elif name in PRESETS:
        base = PRESETS[name]
    else:
        raise HaloVLAConfigError(f"unknown dual_system {name!r}; choose none, custom or one of {sorted(PRESETS)}")
    overrides = dict(overrides or {})
    unknown = set(overrides) - {f.name for f in fields(DualSystemConfig)}
    if unknown:
        raise HaloVLAConfigError(f"unknown dual_system options: {sorted(unknown)}")
    return replace(base, **overrides)
