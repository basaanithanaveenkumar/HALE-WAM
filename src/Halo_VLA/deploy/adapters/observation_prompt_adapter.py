"""Base of the adapters whose graph reads a full observation + prompt (images, ids, states)."""

from __future__ import annotations

from typing import Any

from Halo_VLA.deploy.adapters.halo_adapter_base import HaloAdapterBase
from Halo_VLA.deploy.prompt_io_signature import PromptIOSignature


class ObservationPromptAdapter(HaloAdapterBase):
    """Inputs ``images [B, N, 3, H, W]``, ``input_ids [B, L]``, ``attention_mask [B, L]``,
    ``states [B, S, state_dim]``; ``L`` is dynamic between the template minimum and ``max_seq``.

    Args:
        num_images: frames per observation (one ``<image>`` token each).
        num_states: ``<state>`` slots in the prompt.
        num_action_tokens: ``<halo_action>`` tokens (one action chunk each).
        num_world_video_tokens: ``<halo_world_video>`` tokens the prompt carries.
        opt_seq / max_seq: prompt lengths for tracing / the largest accepted prompt.
    """

    def __init__(
        self,
        num_images: int = 1,
        num_states: int = 1,
        num_action_tokens: int = 1,
        num_world_video_tokens: int = 0,
        opt_seq: int = 64,
        max_seq: int = 256,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        self.num_images = int(num_images)
        self.num_states = int(num_states)
        self.num_action_tokens = int(num_action_tokens)
        self.num_world_video_tokens = int(num_world_video_tokens)
        self.opt_seq = int(opt_seq)
        self.max_seq = int(max_seq)

    def observation_signature(
        self, outputs: list[str], extra_inputs: list[dict[str, Any]] | tuple[()] = ()
    ) -> PromptIOSignature:
        """Signature of an observation + prompt graph.

        Args:
            outputs: names of the graph outputs.
            extra_inputs: further input specs (e.g. noise).

        Returns:
            A :class:`PromptIOSignature` whose prompt template carries the special tokens.
        """
        cfg = self.model_config()
        head = (cfg.image_token_id,) * self.num_images + (cfg.state_token_id,) * self.num_states
        tail = (cfg.world_video_token_id,) * self.num_world_video_tokens + (
            cfg.action_token_id,
        ) * self.num_action_tokens
        special = min(
            cfg.image_token_id, cfg.state_token_id, cfg.action_token_id, cfg.world_video_token_id
        )
        min_seq = 2 + len(head) + len(tail)
        base = self.build_signature(
            {
                "inputs": [
                    {
                        "name": "images",
                        "shape": ["batch", self.num_images, 3, cfg.img_size, cfg.img_size],
                        "dtype": "float32",
                        "value_range": [0.0, 1.0],
                    },
                    {
                        "name": "input_ids",
                        "shape": ["batch", "seq"],
                        "dtype": "int64",
                        "value_range": [3, special],
                    },
                    {
                        "name": "attention_mask",
                        "shape": ["batch", "seq"],
                        "dtype": "int64",
                        "value_range": [1, 2],
                    },
                    {
                        "name": "states",
                        "shape": ["batch", self.num_states, cfg.state_dim],
                        "dtype": "float32",
                        "value_range": [-1.0, 1.0],
                    },
                    *extra_inputs,
                ],
                "outputs": outputs,
                "dims": {
                    "batch": self.batch_dim(),
                    "seq": [
                        min_seq,
                        max(min_seq, self.opt_seq),
                        max(min_seq, self.opt_seq, self.max_seq),
                    ],
                },
            }
        )
        return PromptIOSignature(base.inputs, base.outputs, base.dims, "input_ids", head, tail)
