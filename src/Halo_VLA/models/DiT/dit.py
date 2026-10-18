"""DiT: diffusion Transformer for image-like data (frames, depth, flow)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_block import DiTBlock
from .scalar_timestep_embedder import ScalarTimestepEmbedder

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig


class DiT(nn.Module):
    """Diffusion Transformer for image-like data (frames, depth, flow).

    Patchifies the noisy input, adds (interpolated) learned positions, runs adaLN-Zero blocks
    conditioned on the timestep gated by a context vector, then unpatchifies.

    Args:
        config: settings with the ``dit_*`` fields (``dit_in_channels``, ``dit_out_channels``,
            ``dit_patch_size``, ``dit_hidden_size``, ``dit_num_heads``, ``dit_depth``,
            ``dit_max_resolution``, ``dit_mlp_ratio``, optional ``dit_time_freq_dim``) and ``emb_dim``.
    """

    def __init__(self, config: HaloVLMConfig | SimpleNamespace) -> None:
        super().__init__()
        in_channels = config.dit_in_channels
        out_channels = config.dit_out_channels
        patch_size = config.dit_patch_size
        hidden_size = config.dit_hidden_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = config.dit_num_heads
        self.depth = config.dit_depth

        # Patchify: conv -> linear projection
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, padding=0
        )
        # Number of visual tokens at maximum resolution (in patch units, not pixels)
        self.max_res = config.dit_max_resolution
        num_patches = (self.max_res // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)
        freq_dim = getattr(config, "dit_time_freq_dim", 256)
        self.time_embed = ScalarTimestepEmbedder(hidden_size, frequency_embedding_size=freq_dim)
        # Context embedding: project then scale+shift the time embedding so the
        # per-frame conditioning signal is multiplicatively fused rather than
        # additively swamped by the timestep embedding.
        self.context_embed = nn.Linear(config.emb_dim, hidden_size)
        # Gate: sigmoid(W*ctx) element-wise scales t_emb so context actively modulates it.
        self.context_gate = nn.Linear(config.emb_dim, hidden_size)
        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, self.num_heads, config.dit_mlp_ratio) for _ in range(self.depth)]
        )
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Unpatchify: linear -> conv to reconstruct pixel space
        self.unpatch_embed = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.out_channels = out_channels
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Xavier-initialise linears, then zero the adaLN modulation and unpatchify layers (adaLN-Zero)."""
        # Standard weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Zero-out the last layer of each adaLN modulation for stability (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # context_gate bias → 0 so sigmoid(0)=0.5: equal time/context blend at init
        nn.init.constant_(self.context_gate.bias, 0)
        # Zero-out the unpatch embedding output layer (optional)
        nn.init.constant_(self.unpatch_embed.weight, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the velocity (or noise) for a noisy input.

        Args:
            x: noisy input ``[B, C, H, W]`` (frame, depth, flow).
            t: diffusion / flow time ``[B]`` (int or float).
            context: conditioning ``[B, emb_dim]`` from HaloVLM, or ``None``.

        Returns:
            ``[B, out_C, H, W]``.
        """
        B, C, H, W = x.shape
        # 1. Patchify
        x_patch = self.patch_embed(x)  # [B, hidden_size, H/patch, W/patch]
        ph, pw = x_patch.shape[2], x_patch.shape[3]
        x_patch = x_patch.flatten(2).transpose(1, 2)  # [B, N, hidden_size], N = ph*pw

        # 2. Add positional embedding (interpolate if resolution differs from max)
        max_ph = self.max_res // self.patch_size
        if ph != max_ph or pw != max_ph:
            pos_embed = (
                F.interpolate(
                    self.pos_embed.reshape(1, max_ph, max_ph, -1).permute(0, 3, 1, 2),
                    size=(ph, pw),
                    mode="bilinear",
                    align_corners=False,
                )
                .flatten(2)
                .permute(0, 2, 1)
            )
        else:
            pos_embed = self.pos_embed
        x_patch = x_patch + pos_embed

        # 3. Prepare conditioning vector.
        # Gate the time embedding by the context so each frame's conditioning
        # actively modulates (rather than just shifts) the diffusion signal.
        t_emb = self.time_embed(t)  # [B, hidden_size]
        if context is not None:
            gate = torch.sigmoid(self.context_gate(context))  # [B, hidden_size] ∈ (0,1)
            ctx = self.context_embed(context)  # [B, hidden_size]
            c = gate * t_emb + ctx
        else:
            c = t_emb

        # 4. Transformer blocks
        for block in self.blocks:
            x_patch = block(x_patch, c)

        # 5. Final LN and unpatchify
        x_patch = self.final_norm(x_patch)
        x_patch = self.unpatch_embed(x_patch)  # [B, N, patch*patch*out_C]
        # Reshape to spatial image
        x_out = x_patch.reshape(B, ph, pw, self.patch_size, self.patch_size, self.out_channels)
        x_out = x_out.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_out = x_out.reshape(B, self.out_channels, H, W)
        return x_out
