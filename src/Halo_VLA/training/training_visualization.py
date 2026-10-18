"""Periodic training visualisations: per-sample frames, answers, action plots and DiT GIFs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from config import HaloVLMConfig
from visualization.action_plots import action_comparison_image, unnormalise_image
from visualization.diffusion_gifs import sample_dit_with_intermediates, save_diffusion_gif
from visualization.drawing import get_font
from visualization.text_generation import generate_text

# Optional renderers: frames need OpenCV, GIFs need imageio; without them visualisation is skipped.
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


def draw_overlays(
    frame: np.ndarray,
    question: str,
    gt_answer: str,
    pred_answer: str,
    gt_actions: np.ndarray | None,
    pred_actions: np.ndarray | None,
    frame_idx: int,
) -> np.ndarray:
    """Draw the question, ground-truth and predicted answers onto a BGR frame (no-op without OpenCV).

    Args:
        frame: BGR frame; drawn on in place.
        question: task text.
        gt_answer: ground-truth answer.
        pred_answer: model answer (omitted when empty).
        gt_actions: ground-truth actions (unused by the text overlay).
        pred_actions: predicted actions (unused by the text overlay).
        frame_idx: index of the frame (unused by the text overlay).

    Returns:
        The annotated frame.
    """
    if not _CV2_AVAILABLE:
        return frame
    cv2.putText(
        frame, f"Q: {question[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"GT: {gt_answer[:30]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )
    if pred_answer:
        cv2.putText(
            frame,
            f"Pred: {pred_answer[:30]}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return frame


# ---------------------------------------------------------------------------
# Unified training visualization
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_training_visualization(
    model: nn.Module,
    batch: dict[str, Any],
    tokenizer: Any,
    config: HaloVLMConfig,
    device: torch.device,
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
    img_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    img_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> None:
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

    from PIL import Image as PILImage
    from PIL import ImageDraw as PILDraw

    # ── layout ───────────────────────────────────────────────
    HDR_H = 88
    PANEL_H = panel_h
    PANEL_W = panel_w
    COL_SEP = 4
    ROW_SEP = 26
    CW = PANEL_W * 2 + COL_SEP
    CH = HDR_H + PANEL_H + ROW_SEP + PANEL_H

    # RGB colour palette (PIL works in RGB; convert to BGR only at write time).
    C_BG = (14, 16, 28)
    C_MID = (22, 26, 44)
    C_SEP = (60, 68, 104)
    C_TXT = (220, 226, 240)
    C_DIM = (90, 98, 120)
    C_WHT = (255, 255, 255)
    C_GRN = (130, 235, 170)  # GT
    C_CYN = (255, 200, 110)  # prediction (amber)
    C_ORG = (150, 220, 255)  # task / accent blue
    C_ACCENT = (96, 200, 255)

    F_LABEL = get_font(14, bold=True)
    F_BODY = get_font(15)
    F_PANEL = get_font(13, bold=True)
    F_BAND = get_font(13, bold=True)
    F_BADGE = get_font(12, bold=True)

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
        d.line((0, hr + ROW_SEP - 1, CW, hr + ROW_SEP - 1), fill=C_SEP, width=1)
        band_label = "Future Frame Prediction"
        bbox = d.textbbox((0, 0), band_label, font=F_BAND)
        tw = bbox[2] - bbox[0]
        d.text(((CW - tw) // 2, hr + (ROW_SEP - 16) // 2), band_label, font=F_BAND, fill=C_CYN)
        return canvas

    def panel_origin(row: int, col: int) -> tuple[int, int]:
        """Top-left pixel of the panel at grid ``(row, col)``."""
        x0 = col * (PANEL_W + COL_SEP)
        y0 = HDR_H + row * (PANEL_H + ROW_SEP)
        return x0, y0

    def _wrap(text: str, max_chars: int) -> str:
        if not text:
            return ""
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def draw_header(
        canvas: Any,
        task: str,
        gt: str,
        pred_partial: str,
        cursor: bool = False,
        is_correct: bool | None = None,
    ) -> None:
        """Draw the task / ground-truth / prediction rows and the optional correctness badge."""
        d = PILDraw.Draw(canvas, "RGBA")
        max_c = max(20, (CW - 30) // 9)
        tick = "|" if cursor else ""

        # Three labelled rows, each with a bold tag and a value.
        labels = [
            ("TASK", _wrap(task, max_c), C_ORG),
            ("GT", _wrap(gt, max_c), C_GRN),
            ("PRED", _wrap(pred_partial + tick, max_c), C_CYN),
        ]
        y = 10
        for tag, val, val_col in labels:
            d.text((18, y + 1), tag, font=F_LABEL, fill=(0, 0, 0, 200))
            d.text((17, y), tag, font=F_LABEL, fill=C_WHT)
            d.text((78, y + 1), val, font=F_BODY, fill=(0, 0, 0, 200))
            d.text((77, y), val, font=F_BODY, fill=val_col)
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
                d.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=badge_col + (235,))
            except AttributeError:
                d.rectangle((x0, y0, x1, y1), fill=badge_col + (235,))
            d.text((x0 + pad_x, y0 + pad_y - 1), label, font=F_BADGE, fill=(255, 255, 255))

    def place_image(
        canvas: Any,
        img_bgr: np.ndarray,
        row: int,
        col: int,
        label: str,
        color: tuple[int, ...] = C_WHT,
        border: tuple[int, ...] | None = None,
    ) -> None:
        """Paste a BGR image into a grid panel with a label ribbon and optional border."""
        x0, y0 = panel_origin(row, col)
        panel = cv2.resize(img_bgr, (PANEL_W, PANEL_H))
        panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        canvas.paste(PILImage.fromarray(panel_rgb), (x0, y0))

        d = PILDraw.Draw(canvas, "RGBA")
        if border:
            d.rectangle((x0, y0, x0 + PANEL_W - 1, y0 + PANEL_H - 1), outline=border, width=2)
        # Top label ribbon.
        d.rectangle((x0, y0, x0 + PANEL_W, y0 + 24), fill=(0, 0, 0, 160))
        d.text((x0 + 9, y0 + 5), label, font=F_PANEL, fill=color)

    def place_chart(
        canvas: Any,
        chart_bgr: np.ndarray,
        row: int,
        col: int,
        label: str = "",
        color: tuple[int, ...] = C_TXT,
    ) -> None:
        """Paste a chart image into a grid panel with an optional bottom label."""
        x0, y0 = panel_origin(row, col)
        resized = cv2.resize(chart_bgr, (PANEL_W, PANEL_H))
        chart_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas.paste(PILImage.fromarray(chart_rgb), (x0, y0))
        if label:
            d = PILDraw.Draw(canvas, "RGBA")
            d.rectangle((x0, y0 + PANEL_H - 22, x0 + PANEL_W, y0 + PANEL_H), fill=(0, 0, 0, 160))
            d.text((x0 + 9, y0 + PANEL_H - 18), label, font=F_PANEL, fill=color)

    def place_placeholder(canvas: Any, row: int, col: int, label: str) -> None:
        """Fill a grid panel with a hatched placeholder and a label."""
        x0, y0 = panel_origin(row, col)
        d = PILDraw.Draw(canvas, "RGBA")
        d.rectangle((x0, y0, x0 + PANEL_W, y0 + PANEL_H), fill=C_BG)
        # Hatched pattern.
        for offset in range(0, PANEL_W + PANEL_H, 14):
            d.line((x0 + offset, y0, x0, y0 + offset), fill=(28, 32, 50), width=1)
        bbox = d.textbbox((0, 0), label, font=F_LABEL)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(
            ((x0 + (PANEL_W - tw) // 2), y0 + (PANEL_H - th) // 2), label, font=F_LABEL, fill=C_DIM
        )

    def render_action_chart(
        gt_arr: np.ndarray | None, pred_arr: np.ndarray | None, reveal_up_to: int | None = None
    ) -> np.ndarray:
        """Use the polished shared chart renderer from visualize.py."""
        if gt_arr is None and pred_arr is None:
            return np.full((PANEL_H, PANEL_W, 3), (28, 32, 50), dtype=np.uint8)
        if gt_arr is None:
            # action_comparison_image expects a GT array; substitute zeros.
            gt_arr = np.zeros_like(pred_arr)
        return action_comparison_image(
            pred=pred_arr,
            gt=gt_arr,
            width=PANEL_W,
            height=PANEL_H,
            reveal_up_to=reveal_up_to,
            phase_label="Action Trajectory",
        )

    def to_rgb_array(canvas: Any) -> np.ndarray:
        """Convert a PIL canvas to an RGB ``uint8`` array."""
        return np.asarray(canvas.convert("RGB"))

    # ── load batch ───────────────────────────────────────────
    model.eval()
    vis_dir = output_dir / f"vis_epoch{epoch}_step{global_step}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = batch["images"].to(device)
    input_ids_full = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    states = batch["states"].to(device)
    image_mask = batch.get("image_mask")
    gt_future = batch.get("future_frames")
    if gt_future is not None:
        gt_future = gt_future.to(device)

    has_predictor = hasattr(model, "visual_predictor") and model.visual_predictor is not None
    num_predict_cap = (
        getattr(model.visual_predictor, "num_predict_frames", 1) if has_predictor else 0
    )

    B = images.size(0)
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
            enc = tokenizer(
                full_seq[: cut + len(asst_prefix)], return_tensors="pt", add_special_tokens=False
            )
            ids_in = enc["input_ids"].to(device)
            mask_in = enc["attention_mask"].to(device)
        else:
            ids_in = input_ids_full[b : b + 1]
            mask_in = attention_mask[b : b + 1]

        imgs_in = images[b : b + 1]
        states_in = states[b : b + 1]
        img_mask_in = image_mask[b : b + 1] if image_mask is not None else None

        # ── DiT visual future prediction ─────────────────────
        pred_rgb = None
        if has_predictor:
            try:
                _, _, vce, wv = model(
                    images=imgs_in,
                    input_ids=ids_in,
                    attention_mask=mask_in,
                    states=states_in,
                    image_mask=img_mask_in,
                )
                n_frames = min(wv.size(1), num_predict_cap) if wv is not None else num_predict_cap
                preds = model.predict_visual_future(
                    past_rgb=imgs_in,
                    visual_context_emb=vce,
                    num_frames=n_frames,
                    world_video_query_hiddens=wv,
                )
                if preds and "rgb" in preds:
                    pred_rgb = preds["rgb"]
            except Exception as e:  # noqa: BLE001 - visualisation must never kill a training run
                logger.opt(exception=e).warning("Visual pred failed sample {}", b)

        # ── text + action generation ─────────────────────────
        pred_answer = ""
        act_preds = None
        try:
            pred_text, act_preds = generate_text(
                model,
                tokenizer,
                imgs_in,
                ids_in,
                mask_in,
                states_in,
                max_new_tokens=96,
                device=device,
            )
            pred_answer = pred_text.strip()
            for tok in ("<halo_action>", "<halo_world_video>"):
                pred_answer = pred_answer.replace(tok, "").strip()
            if not pred_answer:
                pred_answer = "(no output)"
            pred_answer = pred_answer[:90]
        except Exception as e:  # noqa: BLE001 - visualisation must never kill a training run
            logger.opt(exception=e).warning("Text gen failed sample {}", b)
            pred_answer = "(error)"

        # ── action arrays: full GT traj + predicted chunk ────
        gt_actions_np = None
        pred_actions_np = None
        if batch.get("actions") is not None:
            try:
                gt_actions_np = batch["actions"][b].cpu().float().numpy()  # [T, D]
            except (IndexError, RuntimeError, TypeError) as exc:  # visualisation is best-effort
                logger.debug("no ground-truth actions for sample {}: {}", b, exc)
        if act_preds is not None:
            try:
                # Use only the last chunk (conditioned on the last action token)
                # so the pred length matches chunk_size, not n_act*chunk_size.
                pred_actions_np = act_preds[0, -1].cpu().float().numpy()  # [chunk_size, D]
            except (IndexError, RuntimeError, TypeError) as exc:  # visualisation is best-effort
                logger.debug("no predicted action chunk to plot: {}", exc)

        # ── correctness heuristic for badge ──────────────────
        def _word_overlap(g: str, p: str) -> float:
            gw = set(g.lower().split())
            pw = set(p.lower().split())
            if not gw:
                return 1.0
            return len(gw & pw) / max(len(gw), 1)

        is_correct = _word_overlap(gt_answer, pred_answer) >= 0.5

        # ── pre-render action charts once per sample ─────────
        chart_gt = render_action_chart(gt_actions_np, None)
        chart_both = render_action_chart(gt_actions_np, pred_actions_np)

        last_ctx_bgr = unnormalise_image(images[b, n_imgs - 1].cpu(), img_mean, img_std)
        n_pred = pred_rgb.size(1) if pred_rgb is not None else 0
        n_gt_fut = gt_future.size(1) if gt_future is not None else 0
        pred_words = pred_answer.split()

        gif_frames = []
        gif_durations = []  # ms per GIF frame

        # Timing per GIF frame (ms).
        ms_ctx = max(150, 1000 // max(fps_ctx, 1))
        ms_word = max(80, 1000 // max(fps_text, 1))
        ms_future = max(80, 1000 // max(fps_future, 1))
        # Future-frame window uses its own slower clock so each predicted
        # frame is easy to examine (default 2 fps → 500 ms per hold copy).
        ms_future_frames = max(300, 1000 // max(fps_future_frames, 1))

        # ── Phase 1 : context scroll (slow) ──────────────────
        for i in range(n_imgs):
            c = _new_canvas()
            draw_header(c, task_text, gt_answer, "", is_correct=None)
            ctx_bgr = unnormalise_image(images[b, i].cpu(), img_mean, img_std)
            place_image(c, ctx_bgr, 0, 0, f"Context {i + 1}/{n_imgs}")
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
            draw_header(c, task_text, gt_answer, partial, cursor=not is_last, is_correct=is_correct)
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
                draw_header(c, task_text, gt_answer, pred_answer, is_correct=is_correct)
                place_image(c, last_ctx_bgr, 0, 0, f"Context {n_imgs}/{n_imgs}")
                act_chart = render_action_chart(gt_actions_np, pred_actions_np, reveal_up_to=step_i)
                place_chart(c, act_chart, 0, 1, f"Action  step {step_i}/{n_act_steps}", C_CYN)
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
                place_image(
                    c, gt_bgr, 1, 0, f"GT Future {f + 1}/{n_gt_fut}", color=C_GRN, border=C_GRN
                )
            else:
                place_placeholder(c, 1, 0, f"GT Future {f + 1}")

            if pred_rgb is not None and f < n_pred:
                pf = pred_rgb[0, f].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                pf_bgr = cv2.cvtColor((pf * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                place_image(
                    c, pf_bgr, 1, 1, f"DiT Predicted {f + 1}/{n_pred}", color=C_CYN, border=C_CYN
                )
            else:
                place_placeholder(c, 1, 1, f"DiT Predicted {f + 1}")

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
                    images=imgs_in,
                    input_ids=ids_in,
                    attention_mask=mask_in,
                    states=states_in,
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
                        height=h_in,
                        width=w_in,
                        num_frames=n_frames_d,
                        num_steps=dit_steps,
                        num_snapshots=dit_steps,
                        world_video_query_hiddens=wv_diff,
                    )
                    if snapshots:
                        diff_path = vis_dir / f"sample_{b}_step{global_step}_diffusion.gif"
                        save_diffusion_gif(
                            snapshots,
                            step_times,
                            output_path=str(diff_path),
                            frame_idx=0,
                            sample_idx=0,
                            canvas_w=PANEL_W,
                            canvas_h=PANEL_H,
                            fps=diffusion_fps,
                        )
            except Exception as e:  # noqa: BLE001 - visualisation must never kill a training run
                logger.opt(exception=e).warning("Diffusion GIF failed sample {}", b)

        written += 1

    model.train()
    logger.info("Saved {} visualization GIFs -> {}", written, vis_dir)
