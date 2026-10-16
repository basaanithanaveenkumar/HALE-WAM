"""Dual-system System 2: observation + instruction -> latent tokens in System 1's width."""

from __future__ import annotations

import torch
from torch import nn

from Halo_VLA.deploy.graphs.vlm_sequence_encoder import VLMSequenceEncoder


class System2LatentGraph(nn.Module):
    """``DualSystemVLA.encode_latent`` (System 2 + latent extractor + bridge projector).

    Reproduces ``System2.forward`` at inference: the HiRT-style frame delay
    (``system2_frame_delay``) is applied with a constant index, the learnable ``<ACT>`` queries
    are appended, a middle decoder layer is captured for ``latent: middle_layer``.

    Args:
        dual: the built ``DualSystemVLA``.
    """

    def __init__(self, dual: nn.Module) -> None:
        super().__init__()
        cfg = dual.config
        vlm = dual.system2.model
        layer = None
        if cfg.latent == "middle_layer":
            depth = len(vlm.decoder_layers())
            layer = cfg.middle_layer if cfg.middle_layer is not None else depth // 2
        self.encoder = VLMSequenceEncoder(vlm, capture_layer=layer)
        self.act_queries = dual.system2.act_queries
        self.latent = dual.latent
        self.bridge = dual.bridge
        self.frame_delay = int(cfg.system2_frame_delay)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = images.shape[1]
        if self.frame_delay and n > 1:
            index = [max(i - self.frame_delay, 0) for i in range(n)]
            images = images[:, index]
        b = input_ids.shape[0]
        extra = None if self.act_queries is None else self.act_queries.unsqueeze(0).expand(b, -1, -1)
        hidden, mask, num_prefix, layer_hidden = self.encoder(images, input_ids, attention_mask, states, extra)
        enc = {"hidden": hidden, "mask": mask, "num_prefix": num_prefix, "text_len": input_ids.shape[1],
               "num_extra": 0 if extra is None else extra.shape[1], "layer_hidden": layer_hidden}
        tokens, latent_mask = self.latent(enc)
        return self.bridge(tokens.float()), latent_mask
