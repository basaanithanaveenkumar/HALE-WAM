"""
Train a Dreamer-style agent on top of Halo-VLA's perception.

Two-phase Dreamer loop:

    PHASE 1 (world model):
        frames ──Halo-VLA ViT──▶ obs embeddings o_t
        (o_t, a_t, r_t)  ──▶  fit RSSM (recon + reward + continue + KL)

    PHASE 2 (behaviour in imagination):
        posterior states ──RSSM prior rollout under actor──▶ imagined latents
        critic ← λ-returns of imagined rewards
        actor  ← REINFORCE on imagined advantages + entropy

The world model learns its *own* compact latent dynamics from the (frozen)
Halo-VLA visual features; the actor and critic then train entirely in
imagination — no pixels, no environment steps.

Reward (demonstration): per-step reward = negative distance between each frame
embedding and the final ("goal") frame embedding, i.e. "get closer to the goal
state".  Swap in any ``rl.rewards`` function for a different objective.

Usage:
    python scripts/train_dreamer.py --dataset eo --max_samples 64 --steps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from loguru import logger

from config import HaloVLMConfig
from models.halo_vla import HaloVLM
from rl.rssm import RSSM
from rl.actor_critic import Actor, Critic
from rl.dreamer import world_model_loss, actor_critic_loss


@torch.no_grad()
def encode_frames(model: HaloVLM, images: torch.Tensor) -> torch.Tensor:
    """Encode each frame to a pooled embedding using Halo-VLA's frozen ViT.

    Args:
        images: [B, T, 3, H, W].
    Returns:
        obs_embeds: [B, T, emb_dim].
    """
    B, T = images.shape[:2]
    embs = []
    for t in range(T):
        feat = model.vis_enc(images[:, t])          # [B, P, emb_dim]
        proj = model.image_projector(feat)          # [B, P, emb_dim]
        embs.append(proj.mean(dim=1))               # [B, emb_dim]
    return torch.stack(embs, dim=1)                  # [B, T, emb_dim]


def main():
    args = parse_args()
    device = torch.device(args.device)
    logger.info("Device: {}", device)

    config = HaloVLMConfig(action_dim=args.action_dim, state_dim=args.state_dim)
    model = HaloVLM(config=config).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    obs_dim = config.emb_dim
    rssm = RSSM(action_dim=config.action_dim, obs_embed_dim=obs_dim,
                deter_dim=args.deter_dim, stoch_dim=args.stoch_dim,
                hidden_dim=args.hidden_dim).to(device)
    actor = Actor(rssm.feature_dim, config.action_dim, hidden_dim=args.hidden_dim).to(device)
    critic = Critic(rssm.feature_dim, hidden_dim=args.hidden_dim).to(device)

    wm_opt = Adam(rssm.parameters(), lr=args.wm_lr)
    ac_opt = Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.ac_lr)

    from dataloader.eo_dataset import build_eo_dataloader
    loader = build_eo_dataloader(
        subset=args.subset, split="train", batch_size=args.batch_size,
        num_workers=args.num_workers, img_size=config.img_size,
        max_seq_len=args.max_seq_len, action_dim=config.action_dim,
        state_dim=config.state_dim, num_predict_frames=config.num_visual_predict_frames,
        shuffle=True, max_samples=args.max_samples,
    )
    logger.info("Dataset size: {} | batches: {}", len(loader.dataset), len(loader))

    step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            images = batch["images"].to(device)             # [B, T, 3, H, W]
            T = images.size(1)
            obs_embeds = encode_frames(model, images)       # [B, T, emb_dim]

            # Align actions to the T context frames (subsample / pad the demo).
            actions = batch["actions"].to(device)           # [B, T_act, act_dim]
            actions = actions[:, :T]
            if actions.size(1) < T:
                pad = actions[:, -1:].repeat(1, T - actions.size(1), 1)
                actions = torch.cat([actions, pad], dim=1)

            # Reward: closeness of each frame embedding to the final/goal frame.
            goal = obs_embeds[:, -1:].detach()              # [B, 1, emb_dim]
            rewards = -((obs_embeds - goal) ** 2).mean(dim=-1)  # [B, T]

            # --- Phase 1: world model ---
            wm_loss, wm_metrics, rollout = world_model_loss(
                rssm, obs_embeds, actions, rewards,
                free_nats=args.free_nats, kl_scale=args.kl_scale,
            )
            wm_opt.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), args.grad_clip)
            wm_opt.step()

            # --- Phase 2: actor-critic in imagination ---
            # Seed from the (detached) posterior states of this batch.
            with torch.no_grad():
                seed_state = rssm.observe(obs_embeds, actions)["state"]
            a_loss, c_loss, ac_metrics = actor_critic_loss(
                rssm, actor, critic, seed_state,
                horizon=args.horizon, gamma=args.gamma, lam=args.lam,
                entropy_coef=args.entropy_coef,
            )
            ac_opt.zero_grad()
            (a_loss + c_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), args.grad_clip
            )
            ac_opt.step()

            step += 1
            if step % args.log_every == 0:
                logger.info(
                    "step {} | wm {:.4f} (recon {:.4f} rew {:.4f} kl {:.4f}) | "
                    "actor {:.4f} critic {:.4f} | imag_return {:.4f}",
                    step, wm_metrics["wm_loss"], wm_metrics["recon_loss"],
                    wm_metrics["reward_loss"], wm_metrics["kl"],
                    ac_metrics["actor_loss"], ac_metrics["critic_loss"],
                    ac_metrics["imagined_return"],
                )
            if step >= args.steps:
                logger.info("Reached --steps={}; stopping.", args.steps)
                return
    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Dreamer-style training on Halo-VLA features")
    p.add_argument("--dataset", choices=("eo",), default="eo")
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--state_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    # RSSM / behaviour
    p.add_argument("--deter_dim", type=int, default=128)
    p.add_argument("--stoch_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--horizon", type=int, default=10, help="Imagination horizon H")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--free_nats", type=float, default=1.0)
    p.add_argument("--kl_scale", type=float, default=1.0)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--wm_lr", type=float, default=3e-4)
    p.add_argument("--ac_lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=100.0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    main()
