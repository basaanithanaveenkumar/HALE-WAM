#!/usr/bin/env python3
"""Profile HALE-WAM (Halo-VLA) components: ViT, action decoder, DiT world model, full VLA.

Uses architecture-equivalent stubs so no real weights are needed for profiling.
Component choices reflect the real HALE-WAM building blocks.

Usage:
    python scripts/profile_model.py                          # full VLA stub, CPU
    python scripts/profile_model.py --component vit          # VisTransformer only
    python scripts/profile_model.py --component action       # ActionDecoder MLP
    python scripts/profile_model.py --component decoder      # DecoderTransformer
    python scripts/profile_model.py --device cuda --torch-profiler --trace out/trace.json
    python scripts/profile_model.py --sort flops --json out/profile.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "Halo_VLA"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "HaleBlocks" / "src"))

import torch
import torch.nn as nn

from hale_core.profiling import ModelProfiler


# ---------------------------------------------------------------------------
# Stubs that mirror the real HALE-WAM architecture
# ---------------------------------------------------------------------------

class StubPatchEmbed(nn.Module):
    def __init__(self, img_size=224, p_size=16, in_chans=3, emb_dim=1024):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=p_size, stride=p_size)
        self.num_patches = (img_size // p_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class StubTransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff    = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class StubViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        p_size: int = 16,
        emb_dim: int = 1024,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_dim: int = 2048,
    ):
        super().__init__()
        self.patch_emb = StubPatchEmbed(img_size, p_size, 3, emb_dim)
        n = self.patch_emb.num_patches
        self.pos_embed = nn.Embedding(n, emb_dim)
        self.blocks    = nn.ModuleList([
            StubTransformerBlock(emb_dim, n_heads, mlp_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        p = self.patch_emb(x)
        ids = torch.arange(p.shape[1], device=x.device).unsqueeze(0).expand(b, -1)
        p   = p + self.pos_embed(ids)
        for blk in self.blocks:
            p = blk(p)
        return self.norm(p)


class StubActionDecoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        action_dim: int = 7,
        chunk_size: int = 8,
        hidden_dims: tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        dims = [emb_dim, *hidden_dims, action_dim * chunk_size]
        layers: list[nn.Module] = []
        for i, (di, do) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(di, do))
            if i < len(dims) - 2:
                layers.extend([nn.LayerNorm(do), nn.GELU()])
        self.net       = nn.Sequential(*layers)
        self.action_dim   = action_dim
        self.chunk_size   = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(x.shape[0], self.chunk_size, self.action_dim)


class StubDecoderTransformer(nn.Module):
    def __init__(self, emb_dim: int = 1024, n_layers: int = 6, n_heads: int = 8, mlp_dim: int = 2048):
        super().__init__()
        self.blocks = nn.ModuleList([
            StubTransformerBlock(emb_dim, n_heads, mlp_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).mean(dim=1)  # pool to [B, emb_dim]


class StubHaloVLA(nn.Module):
    """Full VLA stub: ViT → projector → decoder → action head."""

    def __init__(
        self,
        emb_dim: int = 256,
        n_vit_layers: int = 3,
        n_dec_layers: int = 3,
        n_heads: int = 4,
        mlp_dim: int = 512,
    ):
        super().__init__()
        self.vit     = StubViT(emb_dim=emb_dim, n_layers=n_vit_layers, n_heads=n_heads, mlp_dim=mlp_dim)
        self.proj    = nn.Linear(emb_dim, emb_dim)
        self.decoder = StubDecoderTransformer(emb_dim=emb_dim, n_layers=n_dec_layers, n_heads=n_heads, mlp_dim=mlp_dim)
        self.action  = StubActionDecoder(emb_dim=emb_dim)

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        vis  = self.vit(images)          # (B, patches, emb)
        vis  = self.proj(vis)
        ctx  = torch.cat([vis, state.unsqueeze(1)], dim=1)
        h    = self.decoder(ctx)
        return self.action(h)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_vit(args):
    model = StubViT(emb_dim=args.emb_dim, n_layers=args.n_layers, n_heads=args.n_heads, mlp_dim=args.mlp_dim)
    dummy = torch.randn(args.batch_size, 3, 224, 224)
    return model, dummy


def build_action(args):
    model = StubActionDecoder(emb_dim=args.emb_dim)
    dummy = torch.randn(args.batch_size, args.emb_dim)
    return model, dummy


def build_decoder(args):
    model = StubDecoderTransformer(emb_dim=args.emb_dim, n_layers=args.n_layers, n_heads=args.n_heads, mlp_dim=args.mlp_dim)
    seq_len = 196 + 8
    dummy = torch.randn(args.batch_size, seq_len, args.emb_dim)
    return model, dummy


def build_vla(args):
    model = StubHaloVLA(emb_dim=args.emb_dim, n_vit_layers=args.n_layers, n_dec_layers=args.n_layers, n_heads=args.n_heads, mlp_dim=args.mlp_dim)
    images = torch.randn(args.batch_size, 3, 224, 224)
    state  = torch.randn(args.batch_size, args.emb_dim)
    return model, (images, state)


_BUILDERS = {"vit": build_vit, "action": build_action, "decoder": build_decoder, "vla": build_vla}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HALE-WAM profiler")
    p.add_argument("--component",     default="vla", choices=list(_BUILDERS))
    p.add_argument("--device",        default="cpu")
    p.add_argument("--batch-size",    type=int, default=2)
    p.add_argument("--emb-dim",       type=int, default=256,  help="use 1024 for full-scale")
    p.add_argument("--n-heads",       type=int, default=4,    help="use 8 for full-scale")
    p.add_argument("--n-layers",      type=int, default=3,    help="use 6 for full-scale")
    p.add_argument("--mlp-dim",       type=int, default=512,  help="use 2048 for full-scale")
    p.add_argument("--top",           type=int, default=40)
    p.add_argument("--sort",          default="latency", choices=["latency", "flops", "matmuls"])
    p.add_argument("--warmup",        type=int, default=1)
    p.add_argument("--torch-profiler", action="store_true")
    p.add_argument("--trace",         default=None)
    p.add_argument("--json",          default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"\n[profile] HALE-WAM  component={args.component}  device={device}  "
          f"batch={args.batch_size}  emb_dim={args.emb_dim}  n_layers={args.n_layers}")

    model, dummy = _BUILDERS[args.component](args)
    model = model.to(device).eval()

    if isinstance(dummy, tuple):
        dummy = tuple(t.to(device) for t in dummy)
    else:
        dummy = dummy.to(device)

    def _forward():
        return model(*dummy) if isinstance(dummy, tuple) else model(dummy)

    with torch.no_grad():
        for _ in range(args.warmup):
            _forward()

    profiler = ModelProfiler(
        model,
        use_cuda_events=(device.type == "cuda"),
        use_torch_profiler=args.torch_profiler,
        torch_profiler_trace=args.trace,
    )

    with torch.no_grad(), profiler.profile():
        _forward()

    report = profiler.report()
    print("\n" + report.summary_table(top_k=args.top, sort_by=args.sort))

    if args.torch_profiler:
        print("\n── torch.profiler key averages ──")
        print(report.torch_key_averages())

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[profile] JSON → {args.json}")

    if args.trace:
        print(f"[profile] Chrome trace → {args.trace}")


if __name__ == "__main__":
    main()
