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

from dataclasses import dataclass, field
from typing import Any, Literal

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
        """Whether System 2 appends learnable ``<ACT>`` queries to its input."""
        return self.latent in ("act_token", "multi_act")


#: Names that moved to :mod:`dual_system.presets` / :mod:`dual_system.resolution`; resolved lazily
#: so that ``from dual_system.config import PRESETS`` keeps working without an import cycle
#: (those modules import this one).
_MOVED = {
    "PRESETS": "dual_system.presets",
    "DUAL_SYSTEM_PRESETS": "dual_system.presets",
    "NONE": "dual_system.resolution",
    "parse_overrides": "dual_system.resolution",
    "resolve_dual_system_config": "dual_system.resolution",
}


def __getattr__(name: str) -> Any:
    """Backward-compatible access to the relocated preset and resolution helpers."""
    if name in _MOVED:
        import importlib

        return getattr(importlib.import_module(_MOVED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
