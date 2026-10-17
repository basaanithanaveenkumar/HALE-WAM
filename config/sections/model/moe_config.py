"""MoEConfig: deepSeekMoE feed-forwards inside the built-in decoder."""

from __future__ import annotations

from config.sections.base import VLASection


class MoEConfig(VLASection):
    """DeepSeekMoE feed-forwards inside the built-in decoder."""

    OVERRIDES = {"enabled": "use_moe", "hid_scale": "moe_hid_scale", "num_routed_experts": "moe_num_routed_experts",
                 "top_k": "moe_top_k", "num_shared_experts": "moe_num_shared_experts"}

    enabled: bool = True
    hid_scale: float = 1.2
    num_routed_experts: int = 6
    top_k: int = 2
    num_shared_experts: int = 2
