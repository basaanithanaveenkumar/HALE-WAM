"""ActionConfig: action / proprioception spaces and their encoders."""

from __future__ import annotations

from pydantic import Field

from config.sections.base import VLASection

from .flow_config import FlowConfig


class ActionConfig(VLASection):
    """Action / proprioception spaces and their encoders."""

    ARGS = {"action_dim": "action_dim", "state_dim": "state_dim", "chunk_size": "action_chunk_size"}
    OVERRIDES = {
        "hidden_dims": "action_hidden_dims",
        "dropout": "action_dropout",
        "use_layernorm": "action_use_layernorm",
        "state_hidden_dims": "state_hidden_dims",
        "state_dropout": "state_dropout",
    }

    action_dim: int = 32
    state_dim: int = 32
    chunk_size: int = 16
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    use_layernorm: bool = True
    state_hidden_dims: list[int] = Field(default_factory=lambda: [256, 512])
    state_dropout: float = 0.1
    flow: FlowConfig = Field(default_factory=FlowConfig)
