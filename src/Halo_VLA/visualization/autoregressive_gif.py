"""Autoregressive multi-frame DiT rollout rendered as an annotated GIF with a filmstrip."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from visualization.action_plots import (  # noqa: F401 - moved; re-exported for callers
    action_comparison_image,
    tensor_to_display_image,
    unnormalise_image,
)
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
from visualization.text_generation import (  # noqa: F401 - moved; re-exported for callers
    extract_question_answer,
    generate_text,
    load_model,
    simple_match,
)

# Number of denoising snapshots captured per frame.
_AR_DENOISE_SNAPSHOTS = 14


# Duration (ms) for the anchor flash at the start of each frame.
_AR_ANCHOR_MS = 350


def _heun_with_snapshots(
    dit_rgb,
    cond: torch.Tensor,
    height: int,
    width: int,
    rgb_channels: int,
    num_steps: int,
    num_snapshots: int,
    device: torch.device,
):
    """
    Heun 2nd-order ODE integration that also captures intermediate frames.

    Returns:
        final: [1, rgb_channels, H, W]
        snapshots: list of [rgb_channels, H, W] CPU tensors at selected steps
        times: list[float] — t value for each snapshot
    """
    x = torch.randn(1, rgb_channels, height, width, device=device)
    dt = 1.0 / num_steps

    if num_snapshots >= num_steps + 1:
        keep = set(range(num_steps + 1))
    else:
        idxs = np.linspace(0, num_steps, num_snapshots).round().astype(int)
        keep = set(int(i) for i in idxs)

    snapshots, times = [], []

    def _snap(step_idx, x_now):
        snapshots.append(x_now[0].cpu())
        times.append(step_idx * dt)

    if 0 in keep:
        _snap(0, x)

    for i in range(num_steps):
        t_val = i * dt
        t_next_val = min(t_val + dt, 1.0 - 1e-5)
        t_cur = torch.full((1,), t_val, device=device)
        t_nxt = torch.full((1,), t_next_val, device=device)

        v1 = dit_rgb(x, t_cur, cond)
        x_euler = x + v1 * dt
        v2 = dit_rgb(x_euler, t_nxt, cond)
        x = x + (v1 + v2) * 0.5 * dt

        if (i + 1) in keep:
            _snap(i + 1, x)

    return x, snapshots, times


def _filmstrip_bar(
    frames_bgr: List[np.ndarray],
    gt_frames_bgr: List[Optional[np.ndarray]],
    strip_h: int,
    total_w: int,
    active_idx: int,
    num_total: int,
) -> np.ndarray:
    """
    Build a horizontal filmstrip showing observed context frames (cyan border)
    and predicted frames generated so far (green border, gt below each).

    Returns a BGR uint8 array of shape [strip_h, total_w, 3].
    """
    cv2 = _import_cv2()
    bar = np.full((strip_h, total_w, 3), (14, 18, 35), dtype=np.uint8)

    n = len(frames_bgr)
    if n == 0:
        return bar

    thumb_w = min(total_w // max(num_total, 1), strip_h)
    thumb_h = strip_h - 4  # leave a thin top/bottom margin

    for i, (frame, gt) in enumerate(zip(frames_bgr, gt_frames_bgr)):
        x0 = i * thumb_w + 2
        if x0 + thumb_w - 2 > total_w:
            break

        thumb = cv2.resize(frame, (thumb_w - 4, thumb_h - 4),
                           interpolation=cv2.INTER_AREA)
        bar[2:2 + thumb_h - 4, x0 + 2:x0 + thumb_w - 2] = thumb

        # Border colour: cyan for predicted (active or past), white for others
        if i == active_idx:
            border_col = (80, 240, 160)  # bright green — currently generating
        elif gt is not None:
            border_col = (80, 200, 110)  # green — already generated
        else:
            border_col = (96, 200, 255)  # cyan — context / observed

        cv2.rectangle(bar, (x0, 2), (x0 + thumb_w - 2, strip_h - 2),
                      border_col, 1)

        # Place a small GT thumbnail in the bottom-right corner of each
        # predicted slot so the comparison is always visible.
        if gt is not None and i != active_idx:
            gt_w = (thumb_w - 4) // 2
            gt_h = (thumb_h - 4) // 2
            gt_thumb = cv2.resize(gt, (gt_w, gt_h), interpolation=cv2.INTER_AREA)
            gy = 2 + (thumb_h - 4) - gt_h
            gx = x0 + 2 + (thumb_w - 4) - gt_w
            bar[gy:gy + gt_h, gx:gx + gt_w] = gt_thumb
            cv2.rectangle(bar, (gx - 1, gy - 1),
                          (gx + gt_w, gy + gt_h),
                          (255, 210, 100), 1)

    return bar


@torch.no_grad()
def generate_autoregressive_gif(
    model,
    predictor,
    visual_context_emb: torch.Tensor,
    world_video_query_hiddens: Optional[torch.Tensor],
    context_frames_bgr: List[np.ndarray],
    future_frames_bgr: List[np.ndarray],
    context_frames_tensor: torch.Tensor,
    output_path: str,
    canvas_w: int = 728,
    canvas_h: int = 420,
    num_predict: int = 4,
    num_steps: int = 30,
    num_denoise_snapshots: int = _AR_DENOISE_SNAPSHOTS,
    img_mean: Tuple = (0.485, 0.456, 0.406),
    img_std: Tuple = (0.229, 0.224, 0.225),
    fps: int = 12,
):
    """
    Produce an autoregressive future-frame prediction GIF.

    For each predicted frame the GIF shows three sub-phases:
      1. ANCHOR   — the pixel context frame fed to context_frame_enc (highlighted)
      2. DENOISE  — Heun ODE steps: pure noise → sharp predicted RGB frame
      3. REVEAL   — the predicted frame is added to the growing filmstrip;
                    the matching GT frame slides in as a comparison thumbnail

    Each predicted frame is conditioned on the model's own previous prediction
    (i.e. no GT leakage at inference), exactly as ``predict_visual_future`` does.

    Args:
        model: HaloVLM instance (used for config / context_frame_enc access).
        predictor: RGBFramePredictor — the DiT visual predictor.
        visual_context_emb: [1, D] text/visual conditioning from forward().
        world_video_query_hiddens: [1, n_wv, D] or None — per-frame WV queries.
        context_frames_bgr: list of BGR uint8 np.ndarray — the observed frames.
        future_frames_bgr: list of BGR uint8 np.ndarray — ground-truth future frames.
        context_frames_tensor: [T, 3, H, W] normalised tensor — last frame used as
            the initial pixel anchor for context_frame_enc.
        output_path: destination .gif file path.
        canvas_w / canvas_h: frame dimensions for the rendered GIF.
        num_predict: how many future frames to generate.
        num_steps: Heun ODE integration steps.
        num_denoise_snapshots: denoising frames captured per predicted frame.
        img_mean / img_std: ImageNet normalisation parameters.
        fps: GIF playback speed.
    """
    cv2 = _import_cv2()
    from PIL import Image as PILImage

    device = visual_context_emb.device
    rgb_ch = getattr(predictor.config, "rgb_channels", 3)
    H_in = context_frames_tensor.shape[-2]
    W_in = context_frames_tensor.shape[-1]

    # ── State that evolves autoregressively ─────────────────────────────────
    # current_past mirrors halo_vla.predict_visual_future exactly
    current_past_tensor = context_frames_tensor.unsqueeze(0)   # [1, T, 3, H, W]
    filmstrip_bgr: List[np.ndarray] = list(context_frames_bgr)
    filmstrip_gt: List[Optional[np.ndarray]] = [None] * len(context_frames_bgr)
    predicted_bgr: List[np.ndarray] = []
    n_ctx = len(context_frames_bgr)
    n_total_slots = n_ctx + num_predict

    strip_h = max(canvas_h // 5, 60)
    main_h = canvas_h - strip_h - 6    # 6 px divider
    base_dur = int(1000 / fps)

    pil_frames: List = []
    durations: List[int] = []

    def _push(bgr: np.ndarray, dur_ms: int):
        rgb = bgr[:, :, ::-1].copy()
        pil_frames.append(PILImage.fromarray(rgb))
        durations.append(dur_ms)

    def _build_frame(
        main_bgr: np.ndarray,
        phase: str,
        sub: str,
        panel_lines: List[Tuple[str, Tuple[int, int, int], bool]],
        accent: Tuple[int, int, int],
        active_slot: int,
    ) -> np.ndarray:
        """Compose header + main image + filmstrip bar into one canvas."""
        main_resized = cv2.resize(main_bgr, (canvas_w, main_h),
                                  interpolation=cv2.INTER_CUBIC)
        composed = compose_frame(
            main_resized,
            phase_label=phase,
            sub_label=sub,
            panel_lines=panel_lines,
            accent=accent,
            active_phase=3,  # IMAGINE
        )
        strip = _filmstrip_bar(
            filmstrip_bgr, filmstrip_gt,
            strip_h=strip_h,
            total_w=canvas_w,
            active_idx=active_slot,
            num_total=n_total_slots,
        )
        divider = np.full((6, canvas_w, 3), (20, 30, 50), dtype=np.uint8)
        divider[2:4, :] = (96, 200, 255)
        return np.vstack([composed, divider, strip])

    # ── 1. Show all context frames briefly ──────────────────────────────────
    for ci, ctx_bgr in enumerate(context_frames_bgr):
        panel = [
            ("Autoregressive World Prediction", (160, 200, 255), True),
            (f"Context frame {ci + 1}/{n_ctx}", (245, 245, 250), False),
            ("Model input — no prediction yet", (150, 170, 210), False),
        ]
        f = _build_frame(ctx_bgr, "CONTEXT", f"frame {ci+1}/{n_ctx}",
                         panel, (96, 200, 255), ci)
        _push(f, 600)

    # ── 2. Predict frames one at a time ─────────────────────────────────────
    for step in range(num_predict):
        slot_idx = n_ctx + step

        # --- Recompute ctx_emb (ViT-only path for step > 0) ----------------
        if step == 0:
            ctx_emb = visual_context_emb  # [1, D] from full forward()
        else:
            ctx_emb = model.encode_visual_context_from_past_vit(current_past_tensor)

        # --- Per-frame WV query token ----------------------------------------
        wv_step = None
        if (world_video_query_hiddens is not None
                and step < world_video_query_hiddens.size(1)):
            wv_step = world_video_query_hiddens[:, step:step + 1, :]  # [1,1,D]

        # --- Pixel anchor: last frame in current_past -----------------------
        context_frame = current_past_tensor[:, -1]  # [1, 3, H, W]
        anchor_bgr = unnormalise_image(
            context_frame[0], img_mean, img_std
        )
        anchor_display = cv2.resize(anchor_bgr, (canvas_w, main_h),
                                    interpolation=cv2.INTER_CUBIC)

        # Highlight the anchor: draw a glowing green border
        border_t = 8
        anchor_hl = anchor_display.copy()
        anchor_hl[:border_t, :] = (80, 240, 160)
        anchor_hl[-border_t:, :] = (80, 240, 160)
        anchor_hl[:, :border_t] = (80, 240, 160)
        anchor_hl[:, -border_t:] = (80, 240, 160)

        # Phase A: ANCHOR flash — show what context_frame_enc will receive
        for _ in range(3):
            panel = [
                (f"Predicting frame {step + 1}/{num_predict}", (130, 235, 170), True),
                ("ANCHOR — pixel context fed to context_frame_enc", (200, 230, 180), False),
                ("This is the model's last seen/predicted frame", (150, 180, 150), False),
            ]
            f = _build_frame(
                anchor_hl, "IMAGINE", f"frame {step+1}/{num_predict} · ANCHOR",
                panel, (80, 240, 160), slot_idx,
            )
            _push(f, _AR_ANCHOR_MS)

        # --- Build conditioning ------------------------------------------------
        cond = predictor._create_all_frame_contexts(
            ctx_emb,
            num_frames=1,
            world_video_query_hiddens=wv_step,
            frame_offset=step,
            context_frames=context_frame,   # [1, 3, H, W]
            cfg_drop_mask=None,
        )  # [1, D]  (BF = 1*1)

        # Phase B: DENOISE — Heun integration with snapshots
        pred_tensor, snapshots, snap_times = _heun_with_snapshots(
            predictor.dit_rgb, cond, H_in, W_in, rgb_ch,
            num_steps=num_steps,
            num_snapshots=num_denoise_snapshots,
            device=device,
        )

        n_snaps = len(snapshots)
        for si, (snap, t_val) in enumerate(zip(snapshots, snap_times)):
            progress = si / max(n_snaps - 1, 1)
            is_final = si == n_snaps - 1

            # Use proper unnorm for the final snap, min-max for noise blends
            if is_final:
                snap_bgr = unnormalise_image(snap, img_mean, img_std)
            else:
                snap_bgr = tensor_to_display_image(snap)
            snap_bgr = cv2.resize(snap_bgr, (canvas_w, main_h),
                                  interpolation=cv2.INTER_CUBIC)

            noise_col = (
                int(96 + (130 - 96) * progress),
                int(200 - (200 - 235) * progress),
                int(255 - (255 - 170) * progress),
            )
            panel = [
                (f"Predicting frame {step + 1}/{num_predict}", noise_col, True),
                (f"Heun ODE  t = {t_val:.3f}  →  1.000", (220, 220, 255), False),
                (f"Step {si+1}/{n_snaps}  ·  {int(progress*100)}% denoised",
                 (180, 200, 240), False),
            ]
            f = _build_frame(
                snap_bgr,
                "NOISE → FRAME",
                f"frame {step+1}/{num_predict} · t={t_val:.2f}",
                panel,
                noise_col,
                slot_idx,
            )
            dur = int(base_dur * 1.4) if not is_final else 800
            _push(f, dur)

        # Phase C: REVEAL — add prediction to filmstrip, show GT side-by-side
        pred_bgr = unnormalise_image(pred_tensor[0], img_mean, img_std)
        pred_bgr_full = cv2.resize(pred_bgr, (canvas_w, main_h),
                                   interpolation=cv2.INTER_CUBIC)
        filmstrip_bgr.append(cv2.resize(pred_bgr, (strip_h, strip_h)))

        gt_bgr = None
        if step < len(future_frames_bgr):
            gt_bgr = cv2.resize(future_frames_bgr[step], (strip_h, strip_h))
        filmstrip_gt.append(gt_bgr)

        # Split canvas: prediction left, GT right
        half_w = canvas_w // 2
        split = pred_bgr_full.copy()
        if gt_bgr is not None:
            gt_full = cv2.resize(future_frames_bgr[step], (half_w, main_h))
            split[:, half_w:] = gt_full
            # Draw a thin divider between the two halves
            split[:, half_w - 1:half_w + 1] = (255, 255, 255)
            # Label: PRED on left, GT on right
            cv2.putText(split, "PRED", (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                        0.8, (130, 235, 170), 2, cv2.LINE_AA)
            cv2.putText(split, "GT", (half_w + 10, 30), cv2.FONT_HERSHEY_DUPLEX,
                        0.8, (255, 210, 100), 2, cv2.LINE_AA)

        for _ in range(3):
            panel = [
                (f"Frame {step + 1}/{num_predict} predicted", (130, 235, 170), True),
                ("Left: model prediction   Right: ground truth", (200, 220, 200), False),
                ("Pixel anchor = model's own prior frame (no GT leakage)", (150, 200, 160), False),
            ]
            f = _build_frame(
                split, "REVEAL",
                f"frame {step+1}/{num_predict} · REVEAL",
                panel, (130, 235, 170), slot_idx,
            )
            _push(f, 700)

        # --- Feed predicted frame back as the new pixel anchor ---------------
        # Normalise the prediction back to ImageNet space for the ViT/encoder
        pred_norm = pred_tensor[0]  # [3, H, W]  — already in [0,1] approx
        # Clamp and renormalize to match training distribution
        pred_norm = pred_norm.clamp(0, 1)
        mean_t = torch.tensor(img_mean, device=device).view(3, 1, 1)
        std_t = torch.tensor(img_std, device=device).view(3, 1, 1)
        pred_norm = (pred_norm - mean_t) / std_t
        current_past_tensor = torch.cat(
            [current_past_tensor, pred_norm.unsqueeze(0).unsqueeze(0)], dim=1
        )

    # ── 3. Final hold: full filmstrip comparison ─────────────────────────────
    if predicted_bgr:
        last_frame = cv2.resize(predicted_bgr[-1], (canvas_w, main_h))
    elif context_frames_bgr:
        last_frame = cv2.resize(context_frames_bgr[-1], (canvas_w, main_h))
    else:
        last_frame = np.zeros((main_h, canvas_w, 3), dtype=np.uint8)

    panel = [
        (f"Generated {num_predict} future frames", (130, 235, 170), True),
        ("Fully autoregressive — each frame conditions on the previous prediction",
         (200, 230, 200), False),
        ("Green border = predicted   Inset = GT   Cyan = context input",
         (160, 200, 180), False),
    ]
    f = _build_frame(
        last_frame, "IMAGINE",
        f"complete — {num_predict} frames",
        panel, (130, 235, 170), n_total_slots - 1,
    )
    _push(f, 2500)

    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
        )
        logger.info("Saved autoregressive GIF -> {}", output_path)
