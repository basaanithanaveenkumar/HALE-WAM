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

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    import imageio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None  # type: ignore[assignment]
    _IMAGEIO_AVAILABLE = False

import numpy as np


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

def draw_overlays(frame, question, gt_answer, pred_answer, gt_actions, pred_actions, frame_idx):
    if not _CV2_AVAILABLE:
        return frame
    cv2.putText(frame, f"Q: {question[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"GT: {gt_answer[:30]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if pred_answer:
        cv2.putText(frame, f"Pred: {pred_answer[:30]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


# ---------------------------------------------------------------------------
# Unified training visualization
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_training_visualization(
    model,
    batch,
    tokenizer,
    config,
    device,
    output_dir: Path,
    global_step: int,
    epoch: int,
    max_videos: int = 4,
    min_frames: int = 2,
    panel_w: int = 480,
    panel_h: int = 320,
    fps_ctx: int = 6,
    fps_text: int = 8,
    fps_future: int = 10,
    fps_future_frames: int = 2,
    frame_hold: int = 2,
    diffusion_steps: int = 30,
    diffusion_snapshots: int = 24,
    diffusion_fps: int = 12,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
):
    """
    Training-time visualisation GIF — 2x2 panel layout with PIL-rendered text.

        +------------------------------------------------------------------+
        |  Task / GT / Pred (word-by-word reveal)  + correctness badge     |  <- header
        +------------------------------+-----------------------------------+
        |  Context frame  i / N        |  Action trajectory (GT vs Pred)   |
        +------------------------------+-----------------------------------+
        |  GT future frame  f / P      |  DiT predicted frame  f / P       |
        +------------------------------+-----------------------------------+

    Phases:
      1. Context scroll    - top-left cycles through VLA input frames,
                             held slowly so each frame is easy to read.
      2. Autoregressive    - header Pred text revealed one word at a time.
      3. Future reveal     - bottom row shows GT vs DiT predicted futures,
                             action chart switches to GT + Pred overlay.

    A separate noise->image GIF is also written per sample showing the DiT
    diffusion process for the first predicted future frame.
    """
    if not _CV2_AVAILABLE or not _IMAGEIO_AVAILABLE:
        logger.warning("cv2 or imageio not available; skipping visualization")
        return

    from PIL import Image as PILImage, ImageDraw as PILDraw

    # ── layout ───────────────────────────────────────────────
    HDR_H   = 88
    PANEL_H = panel_h
    PANEL_W = panel_w
    COL_SEP = 4
    ROW_SEP = 26
    CW      = PANEL_W * 2 + COL_SEP
    CH      = HDR_H + PANEL_H + ROW_SEP + PANEL_H

    # RGB colour palette (PIL works in RGB; convert to BGR only at write time).
    C_BG       = (14,  16,  28)
    C_MID      = (22,  26,  44)
    C_SEP      = (60,  68,  104)
    C_TXT      = (220, 226, 240)
    C_DIM      = (90,  98,  120)
    C_WHT      = (255, 255, 255)
    C_GRN      = (130, 235, 170)   # GT
    C_CYN      = (255, 200, 110)   # prediction (amber)
    C_ORG      = (150, 220, 255)   # task / accent blue
    C_PINK     = (235, 130, 150)
    C_ACCENT   = (96,  200, 255)

    F_HEAD     = get_font(20, bold=True)
    F_LABEL    = get_font(14, bold=True)
    F_BODY     = get_font(15)
    F_PANEL    = get_font(13, bold=True)
    F_BAND     = get_font(13, bold=True)
    F_BADGE    = get_font(12, bold=True)

    def _new_canvas():
        """Fresh RGB canvas with column/row dividers and a band label."""
        canvas = PILImage.new("RGB", (CW, CH), C_BG)
        d = PILDraw.Draw(canvas, "RGBA")

        # Header strip with a soft horizontal gradient.
        for x in range(CW):
            t = x / max(CW - 1, 1)
            r = int((1 - t) * 18 + t * C_ACCENT[0] * 0.40)
            g = int((1 - t) * 24 + t * C_ACCENT[1] * 0.40)
            b = int((1 - t) * 40 + t * C_ACCENT[2] * 0.50)
            d.line([(x, 0), (x, HDR_H - 1)], fill=(r, g, b))
        d.rectangle((0, 0, 5, HDR_H), fill=C_ACCENT + (255,))
        d.line((0, HDR_H - 1, CW, HDR_H - 1), fill=C_SEP + (220,), width=1)

        # Vertical column divider.
        d.rectangle((PANEL_W, HDR_H, PANEL_W + COL_SEP, CH), fill=C_SEP)

        # Row band between the action chart and the future-frame row.
        hr = HDR_H + PANEL_H
        d.rectangle((0, hr, CW, hr + ROW_SEP), fill=C_MID)
        d.line((0, hr, CW, hr), fill=C_SEP, width=1)
        d.line((0, hr + ROW_SEP - 1, CW, hr + ROW_SEP - 1),
               fill=C_SEP, width=1)
        band_label = "Future Frame Prediction"
        bbox = d.textbbox((0, 0), band_label, font=F_BAND)
        tw = bbox[2] - bbox[0]
        d.text(((CW - tw) // 2, hr + (ROW_SEP - 16) // 2),
               band_label, font=F_BAND, fill=C_CYN)
        return canvas

    def panel_origin(row, col):
        x0 = col * (PANEL_W + COL_SEP)
        y0 = HDR_H + row * (PANEL_H + ROW_SEP)
        return x0, y0

    def _wrap(text: str, max_chars: int) -> str:
        if not text:
            return ""
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def draw_header(canvas, task, gt, pred_partial, cursor=False,
                    is_correct=None):
        d = PILDraw.Draw(canvas, "RGBA")
        max_c = max(20, (CW - 30) // 9)
        tick = "|" if cursor else ""

        # Three labelled rows, each with a bold tag and a value.
        labels = [
            ("TASK", _wrap(task, max_c), C_ORG),
            ("GT",   _wrap(gt,   max_c), C_GRN),
            ("PRED", _wrap(pred_partial + tick, max_c), C_CYN),
        ]
        y = 10
        for tag, val, val_col in labels:
            d.text((18, y + 1), tag, font=F_LABEL, fill=(0, 0, 0, 200))
            d.text((17, y),     tag, font=F_LABEL, fill=C_WHT)
            d.text((78, y + 1), val, font=F_BODY,  fill=(0, 0, 0, 200))
            d.text((77, y),     val, font=F_BODY,  fill=val_col)
            y += 22

        # Correctness badge in the top-right.
        if is_correct is not None:
            label = "CORRECT" if is_correct else "WRONG"
            badge_col = (60, 200, 120) if is_correct else (235, 90, 110)
            bbox = d.textbbox((0, 0), label, font=F_BADGE)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad_x, pad_y = 10, 4
            x1 = CW - 14
            x0 = x1 - tw - pad_x * 2
            y0 = 12
            y1 = y0 + th + pad_y * 2
            try:
                d.rounded_rectangle((x0, y0, x1, y1), radius=10,
                                    fill=badge_col + (235,))
            except AttributeError:
                d.rectangle((x0, y0, x1, y1), fill=badge_col + (235,))
            d.text((x0 + pad_x, y0 + pad_y - 1), label, font=F_BADGE,
                   fill=(255, 255, 255))

    def place_image(canvas, img_bgr, row, col, label,
                    color=C_WHT, border=None):
        x0, y0 = panel_origin(row, col)
        panel = cv2.resize(img_bgr, (PANEL_W, PANEL_H))
        panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        canvas.paste(PILImage.fromarray(panel_rgb), (x0, y0))

        d = PILDraw.Draw(canvas, "RGBA")
        if border:
            d.rectangle((x0, y0, x0 + PANEL_W - 1, y0 + PANEL_H - 1),
                        outline=border, width=2)
        # Top label ribbon.
        d.rectangle((x0, y0, x0 + PANEL_W, y0 + 24),
                    fill=(0, 0, 0, 160))
        d.text((x0 + 9, y0 + 5), label, font=F_PANEL, fill=color)

    def place_chart(canvas, chart_bgr, row, col, label="", color=C_TXT):
        x0, y0 = panel_origin(row, col)
        resized = cv2.resize(chart_bgr, (PANEL_W, PANEL_H))
        chart_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas.paste(PILImage.fromarray(chart_rgb), (x0, y0))
        if label:
            d = PILDraw.Draw(canvas, "RGBA")
            d.rectangle((x0, y0 + PANEL_H - 22, x0 + PANEL_W, y0 + PANEL_H),
                        fill=(0, 0, 0, 160))
            d.text((x0 + 9, y0 + PANEL_H - 18), label, font=F_PANEL,
                   fill=color)

    def place_placeholder(canvas, row, col, label):
        x0, y0 = panel_origin(row, col)
        d = PILDraw.Draw(canvas, "RGBA")
        d.rectangle((x0, y0, x0 + PANEL_W, y0 + PANEL_H), fill=C_BG)
        # Hatched pattern.
        for offset in range(0, PANEL_W + PANEL_H, 14):
            d.line((x0 + offset, y0, x0, y0 + offset),
                   fill=(28, 32, 50), width=1)
        bbox = d.textbbox((0, 0), label, font=F_LABEL)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((x0 + (PANEL_W - tw) // 2), y0 + (PANEL_H - th) // 2),
               label, font=F_LABEL, fill=C_DIM)

    def render_action_chart(gt_arr, pred_arr, reveal_up_to=None):
        """Use the polished shared chart renderer from visualize.py."""
        if gt_arr is None and pred_arr is None:
            return np.full((PANEL_H, PANEL_W, 3), (28, 32, 50), dtype=np.uint8)
        if gt_arr is None:
            # action_comparison_image expects a GT array; substitute zeros.
            gt_arr = np.zeros_like(pred_arr)
        return action_comparison_image(
            pred=pred_arr, gt=gt_arr,
            width=PANEL_W, height=PANEL_H,
            reveal_up_to=reveal_up_to,
            phase_label="Action Trajectory",
        )

    def to_rgb_array(canvas):
        return np.asarray(canvas.convert("RGB"))

    # ── load batch ───────────────────────────────────────────
    model.eval()
    vis_dir = output_dir / f"vis_epoch{epoch}_step{global_step}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    images         = batch["images"].to(device)
    input_ids_full = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    states         = batch["states"].to(device)
    image_mask     = batch.get("image_mask")
    gt_future      = batch.get("future_frames")
    if gt_future is not None:
        gt_future = gt_future.to(device)

    has_predictor   = hasattr(model, "visual_predictor") and model.visual_predictor is not None
    num_predict_cap = getattr(model.visual_predictor, "num_predict_frames", 1) if has_predictor else 0

    B       = images.size(0)
    written = 0

    for b in range(B):
        if written >= max_videos:
            break
        n_imgs = int(image_mask[b].sum().item()) if image_mask is not None else images.size(1)
        if n_imgs < min_frames:
            continue

        # ── decode GT text ───────────────────────────────────
        full_seq = tokenizer.decode(input_ids_full[b].cpu(), skip_special_tokens=False)
        task_text, gt_answer = "", ""
        if "<|im_start|>user" in full_seq:
            blk = full_seq.split("<|im_start|>user")[-1]
            task_text = blk.split("<|im_end|>")[0].strip() if "<|im_end|>" in blk else blk.strip()
        if "<|im_start|>assistant" in full_seq:
            blk = full_seq.split("<|im_start|>assistant")[-1]
            gt_answer = blk.split("<|im_end|>")[0].strip() if "<|im_end|>" in blk else blk.strip()
        for tok in ("<image>", "<state>", "<halo_action>", "<halo_world_video>"):
            task_text = task_text.replace(tok, "").strip()
            gt_answer = gt_answer.replace(tok, "").strip()
        if "Task:" in task_text:
            task_text = task_text.split("Task:")[-1].split("\n")[0].strip()
        task_text = task_text[:90]
        gt_answer = gt_answer[:90]

        # ── build prompt-only ids (strip completed assistant turn) ──
        asst_prefix = "<|im_start|>assistant\n"
        cut = full_seq.rfind(asst_prefix)
        if cut != -1:
            enc     = tokenizer(full_seq[: cut + len(asst_prefix)],
                                return_tensors="pt", add_special_tokens=False)
            ids_in  = enc["input_ids"].to(device)
            mask_in = enc["attention_mask"].to(device)
        else:
            ids_in  = input_ids_full[b : b + 1]
            mask_in = attention_mask[b : b + 1]

        imgs_in     = images[b : b + 1]
        states_in   = states[b : b + 1]
        img_mask_in = image_mask[b : b + 1] if image_mask is not None else None

        # ── DiT visual future prediction ─────────────────────
        pred_rgb = None
        if has_predictor:
            try:
                _, _, vce, wv = model(
                    images=imgs_in, input_ids=ids_in,
                    attention_mask=mask_in, states=states_in,
                    image_mask=img_mask_in,
                )
                n_frames = min(wv.size(1), num_predict_cap) if wv is not None else num_predict_cap
                preds = model.predict_visual_future(
                    past_rgb=imgs_in, visual_context_emb=vce,
                    num_frames=n_frames, world_video_query_hiddens=wv,
                )
                if preds and "rgb" in preds:
                    pred_rgb = preds["rgb"]
            except Exception as e:
                logger.warning("Visual pred failed sample {}: {}", b, e)

        # ── text + action generation ─────────────────────────
        pred_answer = ""
        act_preds   = None
        try:
            pred_text, act_preds = generate_text(
                model, tokenizer, imgs_in, ids_in, mask_in, states_in,
                max_new_tokens=96, device=device,
            )
            pred_answer = pred_text.strip()
            for tok in ("<halo_action>", "<halo_world_video>"):
                pred_answer = pred_answer.replace(tok, "").strip()
            if not pred_answer:
                pred_answer = "(no output)"
            pred_answer = pred_answer[:90]
        except Exception as e:
            logger.warning("Text gen failed sample {}: {}", b, e)
            pred_answer = "(error)"

        # ── action arrays: full GT traj + predicted chunk ────
        gt_actions_np   = None
        pred_actions_np = None
        if batch.get("actions") is not None:
            try:
                gt_actions_np = batch["actions"][b].cpu().float().numpy()  # [T, D]
            except Exception:
                pass
        if act_preds is not None:
            try:
                # Use only the last chunk (conditioned on the last action token)
                # so the pred length matches chunk_size, not n_act*chunk_size.
                pred_actions_np = act_preds[0, -1].cpu().float().numpy()  # [chunk_size, D]
            except Exception:
                pass

        # ── correctness heuristic for badge ──────────────────
        def _word_overlap(g: str, p: str) -> float:
            gw = set(g.lower().split())
            pw = set(p.lower().split())
            if not gw:
                return 1.0
            return len(gw & pw) / max(len(gw), 1)
        is_correct = (_word_overlap(gt_answer, pred_answer) >= 0.5)

        # ── pre-render action charts once per sample ─────────
        chart_gt   = render_action_chart(gt_actions_np, None)
        chart_both = render_action_chart(gt_actions_np, pred_actions_np)

        last_ctx_bgr = unnormalise_image(images[b, n_imgs - 1].cpu(), img_mean, img_std)
        n_pred       = pred_rgb.size(1)  if pred_rgb  is not None else 0
        n_gt_fut     = gt_future.size(1) if gt_future is not None else 0
        pred_words   = pred_answer.split()

        gif_frames    = []
        gif_durations = []          # ms per GIF frame

        # Timing per GIF frame (ms).
        ms_ctx           = max(150, 1000 // max(fps_ctx, 1))
        ms_word          = max( 80, 1000 // max(fps_text, 1))
        ms_future        = max( 80, 1000 // max(fps_future, 1))
        # Future-frame window uses its own slower clock so each predicted
        # frame is easy to examine (default 2 fps → 500 ms per hold copy).
        ms_future_frames = max(300, 1000 // max(fps_future_frames, 1))

        # ── Phase 1 : context scroll (slow) ──────────────────
        for i in range(n_imgs):
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, "", is_correct=None)
            ctx_bgr = unnormalise_image(images[b, i].cpu(), img_mean, img_std)
            place_image(c, ctx_bgr, 0, 0, f"Context {i+1}/{n_imgs}")
            place_chart(c, chart_gt, 0, 1, "Action  GT only", C_GRN)
            place_placeholder(c, 1, 0, "GT Future")
            place_placeholder(c, 1, 1, "DiT Predicted")
            for _ in range(frame_hold):
                gif_frames.append(to_rgb_array(c))
                gif_durations.append(ms_ctx)

        # ── Phase 2 : autoregressive text reveal ─────────────
        for wi in range(max(len(pred_words), 1)):
            partial = " ".join(pred_words[: wi + 1]) if pred_words else pred_answer
            is_last = wi == len(pred_words) - 1
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, partial,
                        cursor=not is_last, is_correct=is_correct)
            place_image(c, last_ctx_bgr, 0, 0, f"Context {n_imgs}/{n_imgs}")
            place_chart(c, chart_gt, 0, 1, "Action  GT only", C_GRN)
            place_placeholder(c, 1, 0, "GT Future")
            place_placeholder(c, 1, 1, "DiT Predicted")
            gif_frames.append(to_rgb_array(c))
            gif_durations.append(ms_word)

        # hold on completed prediction for a beat
        for _ in range(2):
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, pred_answer, is_correct=is_correct)
            place_image(c, last_ctx_bgr, 0, 0, f"Context {n_imgs}/{n_imgs}")
            place_chart(c, chart_gt, 0, 1, "Action  GT only", C_GRN)
            place_placeholder(c, 1, 0, "GT Future")
            place_placeholder(c, 1, 1, "DiT Predicted")
            gif_frames.append(to_rgb_array(c))
            gif_durations.append(ms_ctx)

        # ── Phase 2.5 : action trajectory animated reveal ────
        if pred_actions_np is not None:
            n_act_steps = pred_actions_np.shape[0]
            for step_i in range(1, n_act_steps + 1):
                c = _new_canvas()
                draw_header(c, task_text, gt_answer, pred_answer,
                            is_correct=is_correct)
                place_image(c, last_ctx_bgr, 0, 0,
                            f"Context {n_imgs}/{n_imgs}")
                act_chart = render_action_chart(gt_actions_np, pred_actions_np,
                                               reveal_up_to=step_i)
                place_chart(c, act_chart, 0, 1,
                            f"Action  step {step_i}/{n_act_steps}", C_CYN)
                place_placeholder(c, 1, 0, "GT Future")
                place_placeholder(c, 1, 1, "DiT Predicted")
                gif_frames.append(to_rgb_array(c))
                gif_durations.append(ms_future)

        # ── Phase 3 : future frame reveal ────────────────────
        if pred_actions_np is not None:
            act_lbl = "Action  GT (solid)  /  Pred (dashed)"
            act_col = C_CYN
        else:
            act_lbl = "Action  GT only"
            act_col = C_GRN
        n_future_steps = max(n_pred, n_gt_fut, 1)
        for f in range(n_future_steps):
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, pred_answer, is_correct=is_correct)
            place_image(c, last_ctx_bgr, 0, 0, f"Context {n_imgs}/{n_imgs}")
            place_chart(c, chart_both, 0, 1, act_lbl, act_col)

            if gt_future is not None and f < n_gt_fut:
                gt_bgr = unnormalise_image(gt_future[b, f].cpu(), img_mean, img_std)
                place_image(c, gt_bgr, 1, 0,
                            f"GT Future {f+1}/{n_gt_fut}",
                            color=C_GRN, border=C_GRN)
            else:
                place_placeholder(c, 1, 0, f"GT Future {f+1}")

            if pred_rgb is not None and f < n_pred:
                pf     = pred_rgb[0, f].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                pf_bgr = cv2.cvtColor((pf * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                place_image(c, pf_bgr, 1, 1,
                            f"DiT Predicted {f+1}/{n_pred}",
                            color=C_CYN, border=C_CYN)
            else:
                place_placeholder(c, 1, 1, f"DiT Predicted {f+1}")

            for _ in range(frame_hold):
                gif_frames.append(to_rgb_array(c))
                gif_durations.append(ms_future_frames)

        if not gif_frames:
            continue

        gif_path = vis_dir / f"sample_{b}_step{global_step}.gif"
        imageio.v2.mimsave(str(gif_path), gif_frames, duration=gif_durations, loop=0)

        # ── Separate noise -> image GIF for the first predicted frame ──
        if has_predictor:
            try:
                _, _, vce_diff, wv_diff = model(
                    images=imgs_in, input_ids=ids_in,
                    attention_mask=mask_in, states=states_in,
                    image_mask=img_mask_in,
                )
                if vce_diff is not None:
                    n_frames_d = 1
                    if wv_diff is not None:
                        n_frames_d = max(1, min(wv_diff.size(1), num_predict_cap))
                    h_in, w_in = imgs_in.shape[-2], imgs_in.shape[-1]
                    dit_steps = getattr(config, "dit_num_sample_steps", diffusion_steps)
                    snapshots, step_times = sample_dit_with_intermediates(
                        model.visual_predictor,
                        text_emb=vce_diff,
                        height=h_in, width=w_in,
                        num_frames=n_frames_d,
                        num_steps=dit_steps,
                        num_snapshots=dit_steps,
                        world_video_query_hiddens=wv_diff,
                    )
                    if snapshots:
                        diff_path = vis_dir / f"sample_{b}_step{global_step}_diffusion.gif"
                        save_diffusion_gif(
                            snapshots, step_times,
                            output_path=str(diff_path),
                            frame_idx=0,
                            sample_idx=0,
                            canvas_w=PANEL_W,
                            canvas_h=PANEL_H,
                            fps=diffusion_fps,
                        )
            except Exception as e:
                logger.warning("Diffusion GIF failed sample {}: {}", b, e)

        written += 1

    model.train()
    logger.info("Saved {} visualization GIFs -> {}", written, vis_dir)
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
            raise ValueError("--droid_data_root is required when --dataset droid")
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
            raise ValueError("--moma_data_root is required when --dataset moma")
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
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
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
