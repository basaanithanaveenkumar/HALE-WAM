"""
Train the Halo-VLA action policy with REINFORCE + value baseline.

This script wires the real ``HaloVLM`` model and a dataloader to the RL modules
in ``src/Halo_VLA/rl``.  It demonstrates the full loop end to end:

    observation  ──HaloVLM──▶  action-token hidden state (conditioning ``s``)
                                       │
                          GaussianActionPolicy  ──▶  a ~ π(a|s),  log π(a|s)
                                       │
                                  reward R(a)
                                       │
                       REINFORCE + baseline  ──▶  policy/critic update

Reward
------
For a runnable demonstration the reward is the *negative distance to the
demonstrated (ground-truth) action chunk* — i.e. the policy is rewarded for
matching the expert.  Swap in ``rl.rewards.WorldModelImaginationReward`` to
reward imagined futures from the DiT world model instead.

Usage:
    python scripts/train_reinforce.py --dataset eo --max_samples 64 --steps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

# --- project paths (mirrors scripts/train.py) ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from loguru import logger

from config import HaloVLMConfig
from models.halo_vla import HaloVLM
from rl.policy import GaussianActionPolicy
from rl.value_baseline import ValueBaseline
from rl.reinforce import compute_reinforce_loss
from rl.rewards import negative_distance_reward


def build_loader(args, config):
    """Build a dataloader; only the ``eo`` path needs no local data root."""
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
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_chunk_size=args.action_chunk_size,
    )

    # Frozen Halo-VLA backbone: we only train the RL policy/critic heads here, so
    # the observation embeddings come from the (eval-mode) pre-trained model.
    model = HaloVLM(config=config).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    flat_dim = config.action_chunk_size * config.action_dim
    policy = GaussianActionPolicy(cond_dim=config.emb_dim, action_dim_flat=flat_dim).to(device)
    value = ValueBaseline(cond_dim=config.emb_dim).to(device)

    optim = AdamW(list(policy.parameters()) + list(value.parameters()), lr=args.lr)

    loader = build_loader(args, config)
    logger.info("Dataset size: {} | batches: {}", len(loader.dataset), len(loader))

    step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            # 1. Observation embedding from the frozen backbone.
            cond = GaussianActionPolicy.conditioning_from_halo(model, batch, device=device)

            # 2. Reward target = the demonstrated action chunk, flattened to match
            #    the policy's flat action space.
            actions = batch["actions"].to(device)                # [B, T, act_dim]
            target = actions[:, :config.action_chunk_size].reshape(actions.size(0), -1)
            if target.size(1) < flat_dim:                        # pad short demos
                pad = torch.zeros(target.size(0), flat_dim - target.size(1), device=device)
                target = torch.cat([target, pad], dim=1)
            target = target[:, :flat_dim]

            def reward_fn(a, _t=target):
                return negative_distance_reward(a, _t, offset=args.reward_offset)

            # 3. RL update.
            loss, metrics = compute_reinforce_loss(
                policy, value, cond, reward_fn,
                use_baseline=not args.no_baseline,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value.parameters()), args.grad_clip
            )
            optim.step()

            step += 1
            if step % args.log_every == 0:
                logger.info(
                    "step {} | loss {:.4f} | policy {:.4f} | value {:.4f} | "
                    "reward {:.3f} | entropy {:.3f}",
                    step, metrics["loss"], metrics["policy_loss"],
                    metrics["value_loss"], metrics["reward_mean"], metrics["entropy"],
                )
            if step >= args.steps:
                logger.info("Reached --steps={}; stopping.", args.steps)
                return
    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE + baseline fine-tuning for Halo-VLA")
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
    p.add_argument("--steps", type=int, default=50, help="Stop after N optimizer steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--reward_offset", type=float, default=0.0)
    p.add_argument("--no_baseline", action="store_true", help="Disable the value baseline")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    main()
