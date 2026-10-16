"""Export-friendly re-expression of ``HaloVLM._embed_inputs`` (same weights, same numbers).

``HaloVLM._embed_inputs`` counts placeholders with ``.item()`` and splices state embeddings
in per-sample Python loops - fine for training, untraceable for ``torch.export`` / ONNX / MLX
(data-dependent control flow). This module computes the identical tensor with static-shape
operations only:

* the number of images comes from ``images.shape[1]`` (the prompt holds one ``<image>`` per
  frame, as every Halo-VLA dataloader builds it) and all frames go through the vision encoder
  in **one** batched call instead of a Python loop;
* ``<state>`` slots are filled with a one-hot selection matrix built from a cumulative count
  of the placeholders (``rank = cumsum(is_state)``), i.e. a batched matmul instead of
  ``nonzero`` + item assignment.

Equivalence with the original is checked by the unit tests on random weights.
"""

from __future__ import annotations

import torch
from torch import nn


class StaticPromptEmbedder(nn.Module):
    """``(images, input_ids, states) -> (combined_embeds, num_prefix)``.

    Only the submodules the embedding uses are registered (vision encoder, projector, state
    encoder, text embedding or LLM bridge), so exported graphs do not carry the action / video
    heads as dead weights.

    Args:
        vlm: a built ``HaloVLM`` (its submodules are shared, not copied).
    """

    def __init__(self, vlm: nn.Module) -> None:
        super().__init__()
        cfg = vlm.config
        self.vis_enc = vlm.vis_enc
        self.image_projector = vlm.image_projector
        self.state_encoder = vlm.state_encoder
        self.language = vlm.language
        self.token_emb = vlm.token_emb if vlm.language is None else None
        self.image_token_id = int(cfg.image_token_id)
        self.state_token_id = int(cfg.state_token_id)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                states: torch.Tensor) -> tuple[torch.Tensor, int]:
        b, n_img = images.shape[0], images.shape[1]
        # 1. images: one batched encoder call, then [B, N_img * P, D] in frame order.
        frames = images.reshape(b * n_img, *images.shape[2:])
        img_proj = self.image_projector(self.vis_enc(frames))
        img_proj = img_proj.reshape(b, n_img * img_proj.shape[1], img_proj.shape[2])
        num_prefix = img_proj.shape[1]
        # 2. states: one batched MLP call -> [B, N_state, D].
        n_state = states.shape[1]
        state_embeds = self.state_encoder(states.reshape(b * n_state, states.shape[2]))
        state_embeds = state_embeds.reshape(b, n_state, state_embeds.shape[-1])
        # 3. text embeddings with image / state placeholders zeroed.
        is_image = input_ids == self.image_token_id
        is_state = input_ids == self.state_token_id
        if self.language is not None:
            text = self.language.embed_text(input_ids).to(img_proj.dtype)
            img_proj = self.language.project_in(img_proj)
            state_embeds = self.language.project_in(state_embeds)
        else:
            text = self.token_emb(input_ids)
        keep = (~(is_image | is_state)).unsqueeze(-1).to(text.dtype)
        text = text * keep
        # 4. state injection: k-th <state> token <- k-th state embedding (k < N_state).
        rank = torch.cumsum(is_state.to(torch.int64), dim=1)                       # [B, L]
        slots = torch.arange(1, n_state + 1, device=input_ids.device).view(1, 1, n_state)
        one_hot = ((rank.unsqueeze(-1) == slots) & is_state.unsqueeze(-1)).to(state_embeds.dtype)  # [B, L, S]
        injected = torch.matmul(one_hot, state_embeds)                               # [B, L, D]
        filled = one_hot.sum(-1, keepdim=True)                                       # 1 where injected
        text = text * (1 - filled) + injected
        # 5. image patches prepended to the text / state sequence.
        return torch.cat([img_proj, text], dim=1), num_prefix
