"""Command-line arguments of scripts/train.py (model, data, optimisation, distributed, YAML)."""

import argparse
import sys

import torch
from dual_system import PRESETS as DUAL_PRESETS

from training import (
    RayCliOptions,
    WebDatasetCliOptions,
)
from training.training_visualization import (  # noqa: F401 - moved; re-exported for callers
    draw_overlays,
    generate_training_visualization,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The CLI; with ``--config run.yaml`` the YAML supplies the defaults and flags still win."""
    p = argparse.ArgumentParser(description="Train Halo-VLA")
    p.add_argument("--config", default=None, metavar="YAML",
                   help="Hale-VLM-style run config (configs/*.yaml, `inherits:` supported); command-line "
                        "flags override its values")
    p.add_argument("--model_option", action="append", default=[], metavar="KEY=VALUE",
                   help="set any HaloVLMConfig field, e.g. emb_dim=256, use_moe=false (repeatable)")

    # Data
    p.add_argument(
        "--dataset",
        choices=("eo", "moma", "droid"),
        default="eo",
        help="eo: EO-Data1.5M; moma: local AIRoA MoMa clone; droid: lerobot/droid_100 clone",
    )
    p.add_argument(
        "--droid_data_root",
        default=None,
        help="Path to local lerobot/droid_100 clone. Required for --dataset droid.",
    )
    p.add_argument(
        "--droid_camera",
        default="observation.images.exterior_image_1_left",
        choices=(
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ),
        help="Which DROID camera view to use for image frames.",
    )
    p.add_argument(
        "--moma_data_root",
        default=None,
        help="Path to local airoa-moma clone (episodes.jsonl + videos/). Required for --dataset moma.",
    )
    p.add_argument("--moma_camera", default="head", choices=("hand", "head"))
    p.add_argument(
        "--moma_num_frames",
        type=int,
        default=5,
        help="Context frames per sample fed to the model as input",
    )
    p.add_argument(
        "--num_predict_frames",
        type=int,
        default=5,
        help="Future frames to predict beyond the context window (sets num_visual_predict_frames)",
    )
    p.add_argument("--moma_frame_stride", type=int, default=25)
    p.add_argument("--moma_max_action_len", type=int, default=256)
    p.add_argument(
        "--moma_num_workers",
        type=int,
        default=0,
        help="Video decoding is safest with 0 workers; increase if you use a thread-safe backend",
    )
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--state_dim", type=int, default=32)
    p.add_argument("--action_chunk_size", type=int, default=16)
    p.add_argument(
        "--projector_type",
        default="mlp",
        choices=("mlp", "qformer", "gated_cross_attention"),
        help="Vision -> decoder projector: per-patch MLP, Q-Former, or gated cross-attention.",
    )
    p.add_argument(
        "--num_query_tokens",
        type=int,
        default=None,
        help="Image tokens per image for qformer / gated_cross_attention (default: config value).",
    )
    p.add_argument("--max_samples", type=int, default=2000,
                    help="Limit dataset to N samples for fast loading (None = all)")

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N micro-steps before each optimizer update. "
             "Effective batch = batch_size × grad_accum_steps.",
    )
    p.add_argument("--action_loss_weight", type=float, default=1.0)
    p.add_argument(
        "--visual_loss_weight",
        type=float,
        default=None,
        help="Weight for DiT frame/depth/flow loss (default: config.visual_loss_weight)",
    )
    p.add_argument(
        "--perceptual_weight",
        type=float,
        default=0.1,
        help="Weight for VGG perceptual loss inside vis loss (0 = disabled)",
    )
    p.add_argument(
        "--ssim_weight",
        type=float,
        default=0.4,
        help="Weight for SSIM loss inside vis loss (0 = disabled)",
    )
    p.add_argument(
        "--temporal_weight",
        type=float,
        default=0.2,
        help="Weight for inter-frame temporal smoothness loss (0 = disabled)",
    )

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Logging & checkpoints
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--keep_ckpts", type=int, default=3,
                    help="Keep only the last N checkpoints (0 = keep all)")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a checkpoint (.pt) to resume from. Restores model weights, "
             "optimizer, scheduler, scaler, and global_step. Training continues "
             "from the next epoch after the one stored in the checkpoint.",
    )
    p.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs",
        help="TensorBoard root; logs go under runs/<timestamp>/ unless --tb_run_name is set. Empty string disables.",
    )
    p.add_argument(
        "--tb_run_name",
        type=str,
        default="",
        help="Subfolder under tensorboard_dir (default: timestamp)",
    )
    p.add_argument(
        "--tb_image_every",
        type=int,
        default=0,
        help="Log input frame strip every N steps (0 = off; increases log size)",
    )

    # Visualisation during training
    p.add_argument("--vis_every", type=int, default=10,
                    help="Generate vis videos every N steps (0 = disabled)")
    p.add_argument("--vis_max_videos", type=int, default=4,
                    help="Max videos per visualisation round")
    p.add_argument("--vis_min_frames", type=int, default=3,
                    help="Only visualise samples with >= N frames")
    p.add_argument("--vis_fps_future_frames", type=int, default=2,
                    help="FPS for the future-frame prediction window in training GIFs (lower = slower)")

    # Vision encoder (mirrors Hale-VLM's vision.encoder)
    vis = p.add_argument_group("Vision encoder")
    vis.add_argument("--vision_encoder", default="vit", choices=("vit", "siglip", "clip", "local"),
                     help="vit: built-in ViT; siglip/clip: pretrained Hub tower; local: hale_core.nn.vision backbone")
    vis.add_argument("--vision_model_id", default=None, help="siglip/clip repo (default: *-base-patch16-224)")
    vis.add_argument("--vision_backbone", default="dino_vit_small", help="backbone name for --vision_encoder local")
    vis.add_argument("--vision_freeze", default=None, action=argparse.BooleanOptionalAction,
                     help="freeze the encoder (default: frozen for pretrained encoders, trained for vit)")
    vis.add_argument("--vision_freeze_up_to_stage", type=int, default=-1, help="local: freeze stages 0..N")
    vis.add_argument("--vision_pretrained_path", default=None, help="local: backbone checkpoint")

    # Validation (mirrors Hale-VLM's evaluator): held-out losses + action metrics
    val = p.add_argument_group("Validation")
    val.add_argument("--val_fraction", type=float, default=0.0,
                     help="hold out this share of the training samples for validation (0 = off)")
    val.add_argument("--val_seed", type=int, default=0, help="seed of the held-out permutation")
    val.add_argument("--eval_every", type=int, default=1, help="validate every N epochs")
    val.add_argument("--eval_max_batches", type=int, default=None, help="cap validation batches")
    val.add_argument("--eval_flow_steps", type=int, default=None, help="Euler steps for action sampling")

    # Dual-system VLA (src/Halo_VLA/dual_system, docs/dual_system.md)
    ds = p.add_argument_group("Dual-system VLA")
    ds.add_argument("--dual_system", default="none", choices=("none", "custom", *sorted(DUAL_PRESETS)),
                    help="none: single-system Halo-VLA; a preset / custom: HaloVLM becomes System 2 of a "
                         "slow/fast dual system")
    ds.add_argument("--dual_system_option", action="append", default=[], metavar="KEY=VALUE",
                    help="override a dual-system setting, e.g. latent=pooled, policy=diffusion, async_steps=4 "
                         "(repeatable)")

    # Language backbone (optional; needs HaleBlocks' finetune extra: transformers + peft)
    llm = p.add_argument_group("LLM backbone (optional, replaces the built-in decoder)")
    llm.add_argument("--llm_backbone", default=None,
                     help="registered fine-tunable LLM (hale_core.nn.llm.list_llm_backbones()), e.g. "
                          "smollm2-360m, qwen3-1.7b, smollm3-3b, qwen3.5-4b-base, gemma4-e4b, ministral3-3b, "
                          "cosmos-reason2-2b, lfm2.5-vl-1.6b, hale-autoregressive")
    llm.add_argument("--llm_model_id", default=None, help="override the registered Hub repo / local path")
    llm.add_argument("--llm_checkpoint", default=None, help="HaloDiffusionLLM .pt for hale-* backbones")
    llm.add_argument("--llm_finetune", default="lora", choices=("lora", "full", "frozen"))
    llm.add_argument("--llm_lora_r", type=int, default=16)
    llm.add_argument("--llm_lora_alpha", type=int, default=32)
    llm.add_argument("--llm_lora_dropout", type=float, default=0.05)
    llm.add_argument("--llm_freeze_embeddings", action="store_true",
                     help="keep token embeddings / LM head frozen (default: trained, so the VLA's new "
                          "special tokens are learned)")
    llm.add_argument("--llm_dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    llm.add_argument("--llm_gradient_checkpointing", action="store_true")
    llm.add_argument("--llm_attn_implementation", default="sdpa",
                     help="sdpa / eager / flash_attention_2 (Gemma: eager)")

    # Distributed launch and data format (both optional; need HaleBlocks extras)
    RayCliOptions.add_arguments(p)
    WebDatasetCliOptions.add_arguments(p)

    known, _ = p.parse_known_args(argv)
    if known.config:
        from config.run_config import load_vla_config  # lazy: needs HaleBlocks (pydantic schemas)

        run = load_vla_config(known.config)
        p.set_defaults(**run.train_arguments(), model_overrides=run.model_overrides(), run_variant=run.variant)
    args = p.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)  # recorded next to the checkpoints
    return args
