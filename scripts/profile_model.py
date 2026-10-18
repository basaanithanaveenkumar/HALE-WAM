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

from hale_core.profiling import ModelProfiler
from loguru import logger
from profiling import PROFILE_COMPONENTS, ProfileComponentBuilder



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="HALE-WAM profiler")
    p.add_argument("--component",     default="vla", choices=list(PROFILE_COMPONENTS))
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
    """Profile the chosen component and print the report to stdout."""
    args = parse_args()
    device = torch.device(args.device)

    # The profile report is this command's output (stdout); progress and file paths go to loguru.
    print(f"\n[profile] HALE-WAM  component={args.component}  device={device}  "
          f"batch={args.batch_size}  emb_dim={args.emb_dim}  n_layers={args.n_layers}")

    builder = ProfileComponentBuilder(batch_size=args.batch_size, emb_dim=args.emb_dim,
                                      n_layers=args.n_layers, n_heads=args.n_heads, mlp_dim=args.mlp_dim)
    model, dummy = builder.build(args.component)
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
        logger.info("profile JSON written to {}", args.json)

    if args.trace:
        logger.info("Chrome trace written to {}", args.trace)


if __name__ == "__main__":
    main()
