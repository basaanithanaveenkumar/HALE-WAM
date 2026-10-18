"""
Visualization script for Halo-VLA.

Creates polished MP4 + GIF visualisations from dataset samples (>= 3 frames).
Each video has three phases:
    1. CONTEXT  — input frames played slowly with the question/GT overlay.
    2. ANSWER   — token-by-token reveal of the model's predicted answer.
    3. ACTION   — animated GT vs predicted action trajectory plot.

When a model checkpoint is supplied and the DiT visual predictor is available,
a *separate* GIF is written per sample showing the diffusion noise -> image
conversion for one predicted future frame.

Usage:
    # Dataset only (no model, GT-only video)
    python scripts/visualize.py --output_dir vis_out --num_samples 20

    # With model checkpoint (text + action + diffusion GIF)
    python scripts/visualize.py --checkpoint checkpoints/halo_vla_epoch5.pt \
        --output_dir vis_out --num_samples 20

    # Minimum frame filter
    python scripts/visualize.py --min_frames 5 --num_samples 10
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "Halo_VLA"))

from config import HaloVLMConfig
from dataloader.eo_dataset import EODataset, EODatasetConfig

from loguru import logger
from visualization.drawing import (  # noqa: F401 - moved; re-exported for callers
    _FONT_CACHE,
    _FONT_CANDIDATES_BOLD,
    _FONT_CANDIDATES_REGULAR,
    _PRAI_BAR_H,
    _PRAI_PHASES,
    _bgr_to_pil,
    _import_cv2,
    _import_matplotlib,
    _import_pil,
    _pil_to_bgr,
    add_divider,
    compose_frame,
    draw_correctness_badge,
    draw_header,
    draw_panel,
    draw_prai_bar,
    get_font,
    wrap_text,
)
from visualization.action_plots import (  # noqa: F401 - moved; re-exported for callers
    action_comparison_image,
    tensor_to_display_image,
    unnormalise_image,
)
from visualization.text_generation import (  # noqa: F401 - moved; re-exported for callers
    extract_question_answer,
    generate_text,
    load_model,
    simple_match,
)
from visualization.diffusion_gifs import (  # noqa: F401 - moved; re-exported for callers
    _AR_ANCHOR_MS,
    _AR_DENOISE_SNAPSHOTS,
    _PHASE_LABEL_TO_PRAI,
    _filmstrip_bar,
    _heun_with_snapshots,
    generate_autoregressive_gif,
    sample_dit_with_intermediates,
    save_diffusion_gif,
)
from visualization.video_writer import (  # noqa: F401 - moved; re-exported for callers
    create_video,
)




















































# ---------------------------------------------------------------------------
# Autoregressive future-frame prediction GIF
#
# Visual story per frame:
#   • "ANCHOR"  — flash the pixel context fed to context_frame_enc
#   • "DENOISE" — Heun ODE steps: noise gradually sharpens into the frame
#   • "REVEAL"  — predicted frame snaps onto the growing filmstrip
#   • "GT"      — ground-truth frame slides in beside the prediction
#
# Each predicted frame is conditioned on the model's own previous prediction
# (not the GT frame), mirroring predict_visual_future exactly.
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Render dataset samples (and, with a checkpoint, predictions) as annotated videos."""
    cv2 = _import_cv2()
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = EODatasetConfig(
        subset=args.subset,
        split="train",
        img_size=args.img_size,
        max_seq_len=args.max_seq_len,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
    )
    dataset = EODataset(config=ds_cfg)
    tokenizer = dataset.tokenizer

    model, config = None, None
    if args.checkpoint:
        model, config = load_model(args.checkpoint, device)

    hf_ds = dataset.dataset
    n_total = len(hf_ds)
    written = 0

    logger.info(
        "Scanning {} samples for >= {} frames (max {} videos) ...",
        n_total, args.min_frames, args.num_samples,
    )

    for sample_idx in range(n_total):
        if written >= args.num_samples:
            break

        sample = hf_ds[sample_idx]

        raw_imgs = sample.get("image")
        if raw_imgs is None:
            continue
        if not isinstance(raw_imgs, (list, tuple)):
            raw_imgs = [raw_imgs]
        n_frames = len(raw_imgs)
        if n_frames < args.min_frames:
            continue

        question, gt_answer = extract_question_answer(sample)
        if not question and not gt_answer:
            continue

        images_tensor = dataset._process_images(sample)
        N_img = images_tensor.size(0)

        bgr_frames = [
            unnormalise_image(images_tensor[i], ds_cfg.img_mean, ds_cfg.img_std)
            for i in range(N_img)
        ]

        canvas_h, canvas_w = args.canvas_h, args.canvas_w
        bgr_frames = [cv2.resize(f, (canvas_w, canvas_h)) for f in bgr_frames]

        gt_actions_t, action_mask_t = dataset._process_actions(sample)
        gt_actions = gt_actions_t.numpy() if gt_actions_t.numel() > 0 else None

        pred_answer = None
        pred_actions = None
        diffusion_snapshots = None
        diffusion_times = None

        if model is not None:
            input_ids, attn_mask, _ = dataset._process_conversation(sample, pair_idx=0)
            states_t, _ = dataset._process_states(sample)

            imgs_in = images_tensor.unsqueeze(0).to(device)
            ids_in = input_ids.unsqueeze(0).to(device)
            mask_in = attn_mask.unsqueeze(0).to(device)
            if states_t.numel() > 0:
                states_in = states_t.unsqueeze(0).to(device)
            else:
                states_in = torch.zeros(1, 1, config.state_dim, device=device)

            asst_prefix = "<|im_start|>assistant\n"
            full_text = tokenizer.decode(ids_in[0].cpu(), skip_special_tokens=False)
            cut = full_text.rfind(asst_prefix)
            if cut != -1:
                prompt_text = full_text[: cut + len(asst_prefix)]
                enc = tokenizer(prompt_text, return_tensors="pt",
                                add_special_tokens=False)
                ids_in = enc["input_ids"].to(device)
                mask_in = enc["attention_mask"].to(device)

            pred_text, act_preds = generate_text(
                model, tokenizer, imgs_in, ids_in, mask_in, states_in,
                max_new_tokens=args.max_new_tokens, device=device,
            )
            pred_answer = pred_text.strip()

            if act_preds is not None:
                pred_actions = act_preds[0].view(-1, act_preds.size(-1)).cpu().numpy()

            # --- Diffusion + autoregressive GIFs ---
            visual_predictor = getattr(model, "visual_predictor", None)
            if visual_predictor is not None:
                try:
                    _, _, vce, wv = model(
                        images=imgs_in, input_ids=ids_in,
                        attention_mask=mask_in, states=states_in,
                    )
                    if vce is not None:
                        n_predict = 1
                        if wv is not None:
                            n_predict = min(wv.size(1), visual_predictor.num_predict_frames)
                            n_predict = max(n_predict, 1)
                        # DiT operates at the same H/W as input frames.
                        h_in, w_in = imgs_in.shape[-2], imgs_in.shape[-1]
                        diffusion_snapshots, diffusion_times = sample_dit_with_intermediates(
                            visual_predictor,
                            text_emb=vce,
                            height=h_in, width=w_in,
                            num_frames=n_predict,
                            num_steps=args.diffusion_steps,
                            num_snapshots=args.diffusion_snapshots,
                            world_video_query_hiddens=wv,
                        )
                except Exception as e:
                    logger.warning("Diffusion intermediates failed: {}", e)

            # --- Autoregressive world-prediction GIF ----------------------
            if visual_predictor is not None:
                try:
                    _, _, vce_ar, wv_ar = model(
                        images=imgs_in, input_ids=ids_in,
                        attention_mask=mask_in, states=states_in,
                    )
                    if vce_ar is not None:
                        n_pred_ar = visual_predictor.num_predict_frames
                        if wv_ar is not None:
                            n_pred_ar = min(wv_ar.size(1), n_pred_ar)
                        n_pred_ar = max(n_pred_ar, 1)

                        # Collect GT future frames from sample if available
                        future_rgb_t = sample.get("future_frames", None)
                        gt_future_bgr: List[np.ndarray] = []
                        if future_rgb_t is not None and isinstance(future_rgb_t, torch.Tensor):
                            for fi in range(min(n_pred_ar, future_rgb_t.size(0))):
                                gt_future_bgr.append(
                                    unnormalise_image(future_rgb_t[fi],
                                                     ds_cfg.img_mean, ds_cfg.img_std)
                                )

                        ar_gif_path = str(
                            out_dir / f"sample_{sample_idx:05d}_autoregressive.gif"
                        )
                        generate_autoregressive_gif(
                            model=model,
                            predictor=visual_predictor,
                            visual_context_emb=vce_ar,
                            world_video_query_hiddens=wv_ar,
                            context_frames_bgr=bgr_frames,
                            future_frames_bgr=gt_future_bgr,
                            context_frames_tensor=images_tensor.to(device),
                            output_path=ar_gif_path,
                            canvas_w=args.canvas_w,
                            canvas_h=args.canvas_h,
                            num_predict=n_pred_ar,
                            num_steps=args.ar_steps,
                            img_mean=ds_cfg.img_mean,
                            img_std=ds_cfg.img_std,
                            fps=args.diffusion_fps,
                        )
                except Exception as e:
                    logger.warning("Autoregressive GIF failed: {}", e)

        video_name = f"sample_{sample_idx:05d}_{n_frames}frames.mp4"
        video_path = str(out_dir / video_name)
        create_video(
            frames=bgr_frames,
            question=question,
            gt_answer=gt_answer,
            pred_answer=pred_answer,
            gt_actions=gt_actions,
            pred_actions=pred_actions,
            output_path=video_path,
            fps=args.fps,
            frame_hold=args.frame_hold,
        )

        if diffusion_snapshots is not None and len(diffusion_snapshots) > 0:
            gif_path = str(out_dir / f"sample_{sample_idx:05d}_diffusion.gif")
            save_diffusion_gif(
                diffusion_snapshots, diffusion_times,
                output_path=gif_path,
                frame_idx=0,
                sample_idx=0,
                canvas_w=args.canvas_w,
                canvas_h=args.canvas_h,
                fps=args.diffusion_fps,
                img_mean=ds_cfg.img_mean,
                img_std=ds_cfg.img_std,
            )

        written += 1

    logger.info("Done — {} videos written to {}", written, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Halo-VLA Visualisation")

    # Data
    p.add_argument("--subset", default="interleave-temporal")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--state_dim", type=int, default=32)

    # Filtering
    p.add_argument("--min_frames", type=int, default=3,
                   help="Only visualise samples with >= this many frames")
    p.add_argument("--num_samples", type=int, default=20,
                   help="Max number of videos to produce")

    # Model (optional)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pt checkpoint for generated answers")
    p.add_argument("--max_new_tokens", type=int, default=128)

    # Video
    p.add_argument("--output_dir", default="vis_out")
    p.add_argument("--fps", type=int, default=12, help="Video FPS")
    p.add_argument("--frame_hold", type=float, default=3.0,
                   help="Seconds each context frame is held on screen")
    p.add_argument("--canvas_w", type=int, default=728,
                   help="Video canvas width")
    p.add_argument("--canvas_h", type=int, default=504,
                   help="Video canvas height")

    # Diffusion GIF (noise -> image)
    p.add_argument("--diffusion_steps", type=int, default=30,
                   help="Number of DiT Euler integration steps")
    p.add_argument("--diffusion_snapshots", type=int, default=24,
                   help="Frames captured for the noise->image GIF")
    p.add_argument("--diffusion_fps", type=int, default=12,
                   help="FPS for the noise->image GIF")

    # Autoregressive world-prediction GIF
    p.add_argument("--ar_steps", type=int, default=30,
                   help="Heun ODE steps per frame in the autoregressive GIF")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
