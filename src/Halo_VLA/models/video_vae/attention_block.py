"""AttentionBlock: spatial self-attention (single-head, 1×1 conv QKV) at the bottleneck."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """Spatial self-attention (single-head, 1×1 conv QKV) at the bottleneck."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm  = nn.GroupNorm(32, ch, eps=1e-6, affine=True)
        self.q     = nn.Conv2d(ch, ch, 1)
        self.k     = nn.Conv2d(ch, ch, 1)
        self.v     = nn.Conv2d(ch, ch, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        q   = self.q(h).view(B, C, H * W).permute(0, 2, 1)   # [B, HW, C]
        k   = self.k(h).view(B, C, H * W)                     # [B, C,  HW]
        v   = self.v(h).view(B, C, H * W).permute(0, 2, 1)   # [B, HW, C]
        att = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(att, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(out)
