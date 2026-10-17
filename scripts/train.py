"""
Simple training script for Halo-VLA.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --batch_size 4 --lr 1e-4

Distributed (same script, same losses - see src/Halo_VLA/training/):
    torchrun --nproc_per_node 4 scripts/train.py ...          # DDP on one node
    python scripts/train.py --ray --ray_num_workers 8 --ray_use_gpu ...   # Ray Train (needs hale-blocks[ray])
    python scripts/train.py --webdataset --webdataset_root /fast/shards ...   # any dataset -> WebDataset shards
"""

import argparse
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path so imports work
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from exceptions import CheckpointNotFoundError, HaloVLAConfigError
from config.overrides import apply_model_overrides, parse_model_options
from models.halo_vla import HaloVLM
from dataloader.eo_dataset import build_eo_dataloader
from dataloader.airoa_moma_dataset import build_airoa_moma_dataloader
from dataloader.droid_dataset import build_droid_dataloader
from training import (
    DDPSession,
    DualSystemTrainingStep,
    HaloVLATrainingStep,
    RayCliOptions,
    VLAEvaluator,
    WebDatasetCliOptions,
    launch_train_on_ray,
    split_validation,
)
from dual_system import PRESETS as DUAL_PRESETS, DualSystemVLA, resolve_dual_system_config
from dual_system.config import parse_overrides
from utils import log_module_parameters
from scripts.visualize import (
    unnormalise_image,
    generate_text,
    get_font,
    action_comparison_image,
    sample_dit_with_intermediates,
    save_diffusion_gif,
    tensor_to_display_image,
)

from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

import numpy as np
from training.training_visualization import (  # noqa: F401 - moved; re-exported for callers
    draw_overlays,
    generate_training_visualization,
)
from training.train_arg_parser import (  # noqa: F401 - moved; re-exported for callers
    parse_args,
)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def compute_action_loss(action_preds, action_targets, action_mask):
    """
    MSE loss between predicted and ground-truth actions.

    Args:
        action_preds:   [B, n_act, chunk_size, action_dim] or None
        action_targets: [B, max_T, action_dim]
        action_mask:    [B, max_T]  (1 for real, 0 for pad)
    """
    if action_preds is None:
        return torch.tensor(0.0, device=action_targets.device)

    # Flatten chunk dimension: [B, n_act * chunk_size, action_dim]
    B, n_act, chunk, dim = action_preds.shape
    preds_flat = action_preds.view(B, n_act * chunk, dim)

    # Align lengths (targets may be longer/shorter)
    T = min(preds_flat.size(1), action_targets.size(1))
    preds_flat = preds_flat[:, :T, :]
    targets = action_targets[:, :T, :]
    mask = action_mask[:, :T].unsqueeze(-1).float()  # [B, T, 1]

    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds_flat.device)
    # l1 loss
    loss = ((preds_flat - targets).abs() * mask).sum() / mask.sum() / dim
    return loss



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _latest_ray_checkpoint() -> Path | None:
    """Newest ``halo_vla_epoch*.pt`` in the checkpoint Ray restored this worker from, if any."""
    from hale_core.distributed import RayReporter  # lazy: HaleBlocks is only needed with --ray

    restored = RayReporter.latest_checkpoint_dir()
    if restored is None:
        return None
    ckpts = sorted(Path(restored).glob("halo_vla_epoch*.pt"),
                   key=lambda p: int(p.stem.replace("halo_vla_epoch", "")))
    return ckpts[-1] if ckpts else None


def train(args):
    # ---- Distributed session: single process, torchrun or a Ray Train worker ----
    session = DDPSession.start(args.device)
    device = session.device
    amp_enabled = device.type == "cuda"  # fp16 autocast + loss scaling only exist on CUDA
    if not session.is_main:
        logger.remove()  # only rank 0 prints; every rank still trains
    logger.info("Device: {}  |  rank {}/{}", device, session.rank, session.world_size)

    # ---- Config ----
    config = HaloVLMConfig(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_chunk_size=args.action_chunk_size,
        num_visual_predict_frames=args.num_predict_frames,
        proj_type=args.projector_type,
    )
    if args.num_query_tokens is not None:
        config.qformer_num_queries = args.num_query_tokens
        config.gated_xattn_num_latents = args.num_query_tokens
    if args.visual_loss_weight is not None:
        config.visual_loss_weight = args.visual_loss_weight
    visual_loss_weight = config.visual_loss_weight
    # Vision encoder: built-in ViT, SigLIP / CLIP, or a HaleBlocks backbone (--vision_encoder).
    config.vision_encoder = args.vision_encoder
    config.vision_model_id = args.vision_model_id
    config.vision_backbone = args.vision_backbone
    config.vision_freeze = args.vision_freeze
    config.vision_freeze_up_to_stage = args.vision_freeze_up_to_stage
    config.vision_pretrained_path = args.vision_pretrained_path
    # Optional pretrained, fine-tunable LLM in place of the built-in decoder (--llm_backbone).
    config.llm_backbone = args.llm_backbone
    config.llm_model_id = args.llm_model_id
    config.llm_checkpoint = args.llm_checkpoint
    config.llm_finetune = args.llm_finetune
    config.llm_lora_r, config.llm_lora_alpha = args.llm_lora_r, args.llm_lora_alpha
    config.llm_lora_dropout = args.llm_lora_dropout
    config.llm_lora_train_embeddings = not args.llm_freeze_embeddings
    config.llm_dtype = args.llm_dtype
    config.llm_gradient_checkpointing = args.llm_gradient_checkpointing
    # Dual-system VLA (--dual_system): HaloVLM becomes System 2 of a slow/fast pair.
    config.dual_system = args.dual_system
    config.dual_system_options = parse_overrides(args.dual_system_option) or None
    config.llm_attn_implementation = args.llm_attn_implementation
    # Architecture fields without a flag: from the YAML (--config) and --model_option KEY=VALUE.
    model_overrides = {**getattr(args, "model_overrides", {}), **parse_model_options(args.model_option)}
    if model_overrides:
        apply_model_overrides(config, model_overrides)
        logger.info("HaloVLMConfig overrides: {}", model_overrides)

    # ---- Model ----
    model = HaloVLM(config=config).to(device)
    log_module_parameters(model, model_name="HaloVLM", logger_fn=logger)
    dual_config = resolve_dual_system_config(config.dual_system, config.dual_system_options)
    dual = DualSystemVLA(model, dual_config).to(device) if dual_config is not None else None
    network = dual if dual is not None else model   # what is optimised and checkpointed
    if dual is not None:
        logger.info("dual-system VLA: {} ({})", dual_config.name, dual_config.notes or "custom")
        if args.vis_every > 0:
            logger.warning("training GIFs use Halo-VLA's own action / video heads; disabled for --dual_system")
            args.vis_every = 0
    # With a pretrained LLM its tokenizer (plus the VLA special tokens) defines every token id,
    # so the dataloaders must use it; None keeps the datasets' default cosmo2 tokenizer.
    language = getattr(model, "language", None)
    llm_tokenizer = language.tokenizer if language is not None else None

    # ---- Dataloader ----
    if args.dataset == "eo":
        train_loader = build_eo_dataloader(
            subset=args.subset,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=config.img_size,
            max_seq_len=args.max_seq_len,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            num_predict_frames=args.num_predict_frames,
            shuffle=True,
            max_samples=args.max_samples,
            tokenizer=llm_tokenizer,
        )
    elif args.dataset == "droid":
        if not args.droid_data_root:
            raise HaloVLAConfigError("--droid_data_root is required when --dataset droid")
        train_loader = build_droid_dataloader(
            data_root=args.droid_data_root,
            batch_size=args.batch_size,
            num_workers=args.moma_num_workers,
            img_size=config.img_size,
            max_seq_len=args.max_seq_len,
            max_action_len=args.moma_max_action_len,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            camera=args.droid_camera,
            num_sample_frames=args.moma_num_frames,
            num_predict_frames=args.num_predict_frames,
            frame_stride=args.moma_frame_stride,
            shuffle=True,
            max_samples=args.max_samples,
            tokenizer=llm_tokenizer,
        )
    else:
        if not args.moma_data_root:
            raise HaloVLAConfigError("--moma_data_root is required when --dataset moma")
        moma_workers = args.moma_num_workers
        train_loader = build_airoa_moma_dataloader(
            data_root=args.moma_data_root,
            batch_size=args.batch_size,
            num_workers=moma_workers,
            img_size=config.img_size,
            max_seq_len=args.max_seq_len,
            max_action_len=args.moma_max_action_len,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            num_sample_frames=args.moma_num_frames,
            num_predict_frames=args.num_predict_frames,
            frame_stride=args.moma_frame_stride,
            camera=args.moma_camera,
            shuffle=True,
            max_samples=args.max_samples,
            tokenizer=llm_tokenizer,
        )
    # ---- Grab tokenizer from dataset for debug decoding (before any dataset swap) ----
    tokenizer = train_loader.dataset.tokenizer

    # ---- Optional held-out validation split (fixed permutation: same on every rank / resume) ----
    val_loader = None
    if args.val_fraction > 0:
        train_loader, val_loader = split_validation(train_loader, args.val_fraction, seed=args.val_seed)
        logger.info("validation split: {} train / {} val samples", len(train_loader.dataset), len(val_loader.dataset))
    evaluator = VLAEvaluator(max_batches=args.eval_max_batches, flow_steps=args.eval_flow_steps)

    # ---- Optional WebDataset: write shards once (rank 0), then stream them on every rank ----
    train_loader = WebDatasetCliOptions.prepare_loader(train_loader, args)
    # ---- Each rank reads a disjoint shard of a map-style dataset (no-op single-process) ----
    train_loader = session.shard_loader(train_loader, shuffle=True)
    logger.info("Dataset size: {}  |  Batches per rank: {}", len(train_loader.dataset), len(train_loader))

    # ---- Optimizer & scheduler ----
    # Only trainable parameters: frozen encoders / LLM weights get no optimizer state.
    trainable = (dual.optimizer_parameters() if dual is not None
                 else [p for p in model.parameters() if p.requires_grad])
    logger.info("trainable parameters: {:,} / {:,}", sum(p.numel() for p in trainable),
                sum(p.numel() for p in network.parameters()))
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)))
    scaler = GradScaler(device.type, enabled=amp_enabled)

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.config and session.is_main:
        # Provenance: the YAML as resolved (inheritance applied), plus the final command line.
        from config.run_config import save_resolved_config

        save_resolved_config(args.config, ckpt_dir / "run_config.yaml", argv=getattr(args, "argv", None))

    # ---- Resume from checkpoint ----
    start_epoch = 1
    global_step = 0
    if args.ray and not args.resume:
        # After a worker failure Ray restarts the group with the last reported checkpoint.
        restored = _latest_ray_checkpoint()
        if restored is not None:
            args.resume = str(restored)
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise CheckpointNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Resuming from checkpoint: {}", ckpt_path)
        with torch.serialization.safe_globals([HaloVLMConfig]):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        network.load_state_dict(ckpt["model_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            # Checkpoints from before the trainable-only optimizer hold state for every
            # parameter; keep the weights and restart the optimizer moments.
            logger.warning("optimizer state not restored ({}); continuing with fresh moments", exc)
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        logger.info(
            "Resumed at epoch {} / global_step {}  (training from epoch {})",
            ckpt.get("epoch"), global_step, start_epoch,
        )

    # ---- Model forward + all three losses as one module, DDP-wrapped when distributed ----
    # (the action-flow and DiT losses use parameters outside HaloVLM.forward; wrapping the whole
    # step lets DDP see them, and find_unused_parameters covers batches without future frames)
    if dual is not None:
        dual.set_training_step(global_step)  # OpenHelix pre-alignment stage after a resume
    train_step = session.wrap(
        DualSystemTrainingStep(dual) if dual is not None else HaloVLATrainingStep(
            model,
            action_loss_weight=args.action_loss_weight,
            visual_loss_weight=visual_loss_weight,
            perceptual_weight=args.perceptual_weight,
            ssim_weight=args.ssim_weight,
            temporal_weight=args.temporal_weight,
        ),
        find_unused_parameters=True,
    )
    reporter = None
    if args.ray:
        from hale_core.distributed import RayReporter  # lazy: HaleBlocks is only needed with --ray

        reporter = RayReporter()

    # ---- TensorBoard (rank 0 only) ----
    tb_writer = None
    if args.tensorboard_dir and session.is_main:
        if SummaryWriter is None:
            logger.warning(
                "tensorboard package missing; install with: pip install tensorboard"
            )
        else:
            tb_root = Path(args.tensorboard_dir)
            if args.tb_run_name:
                tb_root = tb_root / args.tb_run_name
            else:
                tb_root = tb_root / datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_writer = SummaryWriter(log_dir=str(tb_root))
            logger.info("TensorBoard log dir: {}  (tensorboard --logdir {})", tb_root, tb_root.parent)

    # ---- Training ----
    model.train()
    # Effective batch = batch_size × grad_accum_steps.
    # grad_accum_steps=1 is identical to the old single-step behaviour.
    accum_steps = max(1, args.grad_accum_steps)
    logger.info(
        "Gradient accumulation: {} micro-steps  →  effective batch {}",
        accum_steps, args.batch_size * accum_steps,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        session.set_epoch(train_loader, epoch)  # new DistributedSampler shuffle every epoch
        epoch_lang_loss = 0.0
        epoch_act_loss = 0.0
        epoch_vis_loss = 0.0
        epoch_total_loss = 0.0
        n_micro = 0   # micro-steps this epoch (for epoch-average denominators)
        t0 = time.time()

        optimizer.zero_grad()   # reset once before accumulation window begins

        for step, batch in enumerate(train_loader, 1):
            # Every tensor of the collated batch to this rank's device (strings etc. stay as is).
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            images = batch["images"]                         # [B, N, 3, H, W]
            input_ids = batch["input_ids"]                   # [B, seq_len]
            attention_mask = batch["attention_mask"]

            # Debug: decode and print input_ids for the first sample in the batch
            if step == 1 and epoch == 1 and session.is_main:
                ids_cpu = input_ids[0].cpu()
                mask_cpu = attention_mask[0].cpu()
                total_tokens = ids_cpu.size(0)
                real_tokens = mask_cpu.sum().item()
                pad_tokens = total_tokens - real_tokens
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                pad_id_count = (ids_cpu == pad_id).sum().item()
                decoded_text = tokenizer.decode(ids_cpu, skip_special_tokens=False)
                logger.info(
                    "=== Debug (sample 0) ===\n"
                    "  Total tokens : {}\n"
                    "  Real tokens  : {}  (attention_mask=1)\n"
                    "  Pad tokens   : {}  (attention_mask=0)\n"
                    "  pad_token_id={} count: {}\n"
                    "  Decoded text:\n{}",
                    total_tokens, real_tokens, pad_tokens,
                    pad_id, pad_id_count, decoded_text,
                )

            is_last_batch = (step == len(train_loader))
            will_step = step % accum_steps == 0 or is_last_batch
            # DDP: skip the gradient all-reduce on accumulation micro-steps; sync on the last one.
            sync = train_step.no_sync() if session.distributed and not will_step else nullcontext()

            # --- Mixed precision forward pass: model + language / action-flow / visual losses ---
            with sync:
                with autocast(device.type, enabled=amp_enabled):
                    out = train_step(batch)
                lang_loss, act_loss, vis_loss = out.lang, out.act, out.vis
                vis_subloss, total_loss = out.vis_sub, out.total

                # --- Backward: scale loss so that accumulated gradients equal the
                #     average over the full effective batch, not the sum.
                scaler.scale(total_loss / accum_steps).backward()

            # Track per-micro-step stats (unscaled loss so epoch avg stays readable)
            n_micro += 1
            epoch_lang_loss  += lang_loss.item()
            epoch_act_loss   += act_loss.item()
            epoch_vis_loss   += vis_loss.item()
            epoch_total_loss += total_loss.item()

            # --- Optimizer step: only every accum_steps micro-steps (or at epoch end) ---
            if will_step:
                # Gradient clipping (must unscale before clipping)
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                if dual is not None:
                    dual.mask_frozen_gradients()   # pre-alignment: System 1 does not move yet

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                if dual is not None:
                    dual.set_training_step(global_step)

                # Logging (keyed to optimizer steps via global_step; rank 0's loader logs)
                if global_step % args.log_every == 0 and session.is_main:
                    lr = optimizer.param_groups[0]["lr"]
                    cfm  = vis_subloss.get("loss_rgb_cfm",        vis_loss).item()
                    perc = vis_subloss.get("loss_rgb_perceptual",  torch.tensor(0.0)).item()
                    ssim = vis_subloss.get("loss_rgb_ssim",        torch.tensor(0.0)).item()
                    temp = vis_subloss.get("loss_rgb_temporal",    torch.tensor(0.0)).item()
                    logger.info(
                        "Epoch {} | Step {}/{} | lang={:.4f}  act={:.4f}  "
                        "vis={:.4f}(cfm={:.4f} perc={:.4f} ssim={:.4f} temp={:.4f})  "
                        "total={:.4f} | lr={:.2e}",
                        epoch, step, len(train_loader),
                        lang_loss.item(), act_loss.item(),
                        vis_loss.item(), cfm, perc, ssim, temp,
                        total_loss.item(), lr,
                    )

                if tb_writer is not None:
                    lr = optimizer.param_groups[0]["lr"]
                    tb_writer.add_scalar("loss/lang", lang_loss.item(), global_step)
                    tb_writer.add_scalar("loss/action_flow", act_loss.item(), global_step)
                    tb_writer.add_scalar("loss/visual_dit", vis_loss.item(), global_step)
                    for k, v in vis_subloss.items():
                        tb_writer.add_scalar(f"loss/{k}", v.item(), global_step)
                    tb_writer.add_scalar("loss/total", total_loss.item(), global_step)
                    tb_writer.add_scalar("optim/lr", lr, global_step)
                    tb_writer.add_scalar("epoch", epoch, global_step)

                    if args.tb_image_every > 0 and global_step % args.tb_image_every == 0:
                        # First sample in batch: strip of frames [N,3,H,W] in [0,1] for TB
                        grid = images[0, : min(8, images.size(1))].detach().cpu()
                        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
                        tb_writer.add_images(
                            "train/input_frames", grid, global_step, dataformats="NCHW"
                        )

                # Visualisation GIFs (rank 0 only; uses the unwrapped model, no collectives)
                if args.vis_every > 0 and global_step % args.vis_every == 0 and session.is_main:
                    generate_training_visualization(
                        model=model, batch=batch, tokenizer=tokenizer, config=config,
                        device=device, output_dir=ckpt_dir, global_step=global_step,
                        epoch=epoch, max_videos=args.vis_max_videos, min_frames=args.vis_min_frames,
                        fps_future_frames=args.vis_fps_future_frames,
                    )



        # End of epoch summary (averaged over all ranks; every rank must take part)
        n = max(n_micro, 1)
        elapsed = time.time() - t0
        avg_lang = session.mean(epoch_lang_loss / n)
        avg_act = session.mean(epoch_act_loss / n)
        avg_vis = session.mean(epoch_vis_loss / n)
        avg_total = session.mean(epoch_total_loss / n)
        logger.info(
            "=== Epoch {} done in {:.1f}s  micro-steps={}  opt-steps={}  | avg lang={:.4f}  act={:.4f}  vis={:.4f}  total={:.4f} ===",
            epoch, elapsed, n_micro, global_step, avg_lang, avg_act, avg_vis, avg_total,
        )

        if tb_writer is not None:
            tb_writer.add_scalar("epoch_avg/lang", avg_lang, epoch)
            tb_writer.add_scalar("epoch_avg/action_flow", avg_act, epoch)
            tb_writer.add_scalar("epoch_avg/visual_dit", avg_vis, epoch)
            tb_writer.add_scalar("epoch_avg/total", avg_total, epoch)

        # Validation (rank 0, unwrapped step: a DDP forward would wait for the other ranks)
        eval_metrics: dict[str, float] = {}
        if val_loader is not None and epoch % args.eval_every == 0:
            if session.is_main:
                plain_step = getattr(train_step, "module", train_step)
                eval_metrics = evaluator.run(plain_step, val_loader, device, amp=amp_enabled)
                logger.info("=== Epoch {} validation | {} ===", epoch,
                            "  ".join(f"{k.removeprefix('val/')}={v:.4f}" for k, v in eval_metrics.items()))
                if tb_writer is not None:
                    for k, v in eval_metrics.items():
                        tb_writer.add_scalar(k, v, epoch)
            session.barrier()

        # Save checkpoint — atomic (write to .tmp then rename) so a disk-full
        # or crash never leaves a corrupted file.  Prune old checkpoints after.
        # Only rank 0 writes; `model` is the unwrapped HaloVLM, so keys carry no DDP prefix.
        save_now = epoch % args.save_every == 0 or epoch == args.epochs
        if save_now and session.is_main:
            ckpt_path = ckpt_dir / f"halo_vla_epoch{epoch}.pt"
            tmp_path  = ckpt_dir / f".tmp_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config,
                },
                tmp_path,
            )
            tmp_path.replace(ckpt_path)   # atomic rename on POSIX
            logger.info("Saved checkpoint → {}", ckpt_path)

            # Prune: keep only the latest `--keep_ckpts` checkpoints
            if args.keep_ckpts > 0:
                old_ckpts = sorted(
                    ckpt_dir.glob("halo_vla_epoch*.pt"),
                    key=lambda p: int(p.stem.replace("halo_vla_epoch", "")),
                )
                for old in old_ckpts[: -args.keep_ckpts]:
                    old.unlink()
                    logger.info("Removed old checkpoint {}", old)

        if reporter is not None:
            # Ray Train: every rank reports; rank 0 attaches the checkpoint directory, which Ray
            # persists under --ray_storage_path and restores from after a worker failure.
            session.barrier()  # the checkpoint is fully written before Ray copies it
            reporter.report(
                {"epoch": epoch, "global_step": global_step, "loss/total": avg_total,
                 "loss/lang": avg_lang, "loss/action_flow": avg_act, "loss/visual_dit": avg_vis,
                 **eval_metrics},
                checkpoint_dir=ckpt_dir if save_now else None,
            )

    logger.info("Training complete.")
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
        logger.info("TensorBoard writer closed.")
    session.barrier()
    session.close()




def main() -> None:
    """Train locally / under torchrun, or launch the same ``train`` on Ray Train with ``--ray``."""
    args = parse_args()
    if args.ray:
        # Workers import this file and call its train(); nothing of this process is pickled.
        result = launch_train_on_ray(f"{Path(__file__).resolve()}:train", args)
        logger.info("Ray run finished: metrics={} checkpoint={}", result.metrics, result.checkpoint_path)
    else:
        train(args)


if __name__ == "__main__":
    main()
