"""
Train the Halo-VLA action policy with GRPO (Group Relative Policy Optimization).

Pipeline:

    observation ──HaloVLM──▶ conditioning ``s``
                                  │
                  for each s: sample G actions a_1..a_G ~ π_old(·|s)
                                  │
                          reward each one R_i
                                  │
              group-relative advantage  Â_i = (R_i − mean)/std
                                  │
              clipped PPO surrogate − β·KL(π_θ‖π_ref)  ──▶ update π_θ

A frozen *reference* policy (deep copy of the initial policy) anchors the KL
penalty and supplies the behaviour distribution π_old for the importance ratio.
Refresh it periodically (``--ref_refresh``) so the trust region tracks the policy.

Usage:
    python scripts/train_grpo.py --dataset eo --max_samples 64 --steps 50 --group_size 8
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from loguru import logger

from config import HaloVLMConfig
from models.halo_vla import HaloVLM
from rl.policy import GaussianActionPolicy
from rl.grpo import compute_grpo_loss
from rl.rewards import negative_distance_reward


def build_loader(args, config):
    from dataloader.eo_dataset import build_eo_dataloader

    return build_eo_dataloader(
        subset=args.subset,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=config.img_size,
        max_seq_len=args.max_seq_len,
        action_dim=config.action_dim,
        state_dim=config.state_dim,
        num_predict_frames=config.num_visual_predict_frames,
        shuffle=True,
        max_samples=args.max_samples,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    logger.info("Device: {}", device)

    config = HaloVLMConfig(
        action_dim=args.action_dim, state_dim=args.state_dim,
        action_chunk_size=args.action_chunk_size,
    )

    model = HaloVLM(config=config).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    flat_dim = config.action_chunk_size * config.action_dim
    policy = GaussianActionPolicy(cond_dim=config.emb_dim, action_dim_flat=flat_dim).to(device)
    # Frozen reference / behaviour policy (KL anchor).
    ref_policy = copy.deepcopy(policy).to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optim = AdamW(policy.parameters(), lr=args.lr)
    loader = build_loader(args, config)
    logger.info("Dataset size: {} | batches: {}", len(loader.dataset), len(loader))

    step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            cond = GaussianActionPolicy.conditioning_from_halo(model, batch, device=device)

            actions = batch["actions"].to(device)
            target = actions[:, :config.action_chunk_size].reshape(actions.size(0), -1)
            if target.size(1) < flat_dim:
                pad = torch.zeros(target.size(0), flat_dim - target.size(1), device=device)
                target = torch.cat([target, pad], dim=1)
            target = target[:, :flat_dim]

            # reward_fn must broadcast the per-state target across the G group
            # samples; group i shares target row i.
            def reward_fn(a, _t=target, _g=args.group_size):
                tgt = _t.repeat_interleave(_g, dim=0)
                return negative_distance_reward(a, tgt, offset=args.reward_offset)

            loss, metrics = compute_grpo_loss(
                policy, ref_policy, cond, reward_fn,
                group_size=args.group_size, clip_eps=args.clip_eps, kl_beta=args.kl_beta,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optim.step()

            step += 1
            if step % args.ref_refresh == 0:
                ref_policy.load_state_dict(policy.state_dict())
            if step % args.log_every == 0:
                logger.info(
                    "step {} | loss {:.4f} | surrogate {:.4f} | kl {:.5f} | "
                    "ratio {:.3f} | reward {:.3f}",
                    step, metrics["loss"], metrics["surrogate"], metrics["kl"],
                    metrics["ratio_mean"], metrics["reward_mean"],
                )
            if step >= args.steps:
                logger.info("Reached --steps={}; stopping.", args.steps)
                return
    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="GRPO fine-tuning for Halo-VLA")
    p.add_argument("--dataset", choices=("eo",), default="eo")
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--state_dim", type=int, default=32)
    p.add_argument("--action_chunk_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--group_size", type=int, default=8, help="Samples G per state")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.04)
    p.add_argument("--ref_refresh", type=int, default=20,
                   help="Sync the reference policy to the current policy every N steps")
    p.add_argument("--reward_offset", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    main()
