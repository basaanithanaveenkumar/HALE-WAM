"""FlowConfig: flow-matching action decoder."""

from __future__ import annotations

from config.sections.base import VLASection


class FlowConfig(VLASection):
    """Flow-matching action decoder."""

    OVERRIDES = {
        "hidden_dim": "flow_hidden_dim",
        "time_embed_dim": "flow_time_embed_dim",
        "num_ode_steps": "flow_num_ode_steps",
    }

    hidden_dim: int = 1024
    time_embed_dim: int = 128
    num_ode_steps: int = 24
