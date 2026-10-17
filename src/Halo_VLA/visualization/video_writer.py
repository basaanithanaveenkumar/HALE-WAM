"""Render a dataset walk-through video with predictions, answers and action plots."""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from visualization.action_plots import (  # noqa: F401 - moved; re-exported for callers
    action_comparison_image,
    tensor_to_display_image,
    unnormalise_image,
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


# ---------------------------------------------------------------------------
# Main video writer
# ---------------------------------------------------------------------------
def create_video(
    frames: List[np.ndarray],
    question: str,
    gt_answer: str,
    pred_answer: Optional[str],
    gt_actions: Optional[np.ndarray],
    pred_actions: Optional[np.ndarray],
    output_path: str,
    fps: int = 12,
    frame_hold: float = 3.0,
    save_gif: bool = True,
):
    """
    Write a polished visualisation for a single sample.

    PHASE 1 (CONTEXT) — each input frame is shown for ``frame_hold`` seconds
        so it's easy to read. Header shows phase + frame counter, bottom
        panel shows the question and GT answer.

    PHASE 2 (ANSWER) — model prediction streams in word-by-word with a typing
        cursor. A correctness badge appears in the top-right.

    PHASE 3 (ACTION) — animated GT vs predicted action trajectory chart, with
        a moving cursor at the currently revealed step.
    """
    cv2 = _import_cv2()

    if not frames:
        logger.warning("No frames to write for {}", output_path)
        return

    H, W = frames[0].shape[:2]

    has_actions = gt_actions is not None and gt_actions.size > 0
    gt_act_traj, pred_act_traj = None, None
    if has_actions:
        gt_act_traj = gt_actions if gt_actions.ndim >= 2 else gt_actions[np.newaxis, :]
        if pred_actions is not None and pred_actions.size > 0:
            pred_act_traj = pred_actions if pred_actions.ndim >= 2 else pred_actions[np.newaxis, :]
            min_d = min(gt_act_traj.shape[-1], pred_act_traj.shape[-1])
            gt_act_traj = gt_act_traj[:, :min_d]
            pred_act_traj = pred_act_traj[:, :min_d]

    chart_h = 260 if has_actions else 0
    divider_h = 6 if has_actions else 0
    total_H = H + divider_h + chart_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, total_H))
    if not writer.isOpened():
        logger.error("Cannot open video writer for {}", output_path)
        return

    is_correct = None
    if pred_answer is not None:
        is_correct = simple_match(gt_answer, pred_answer)

    composed_frames: List[np.ndarray] = []
    frame_durations_ms: List[int] = []

    def _stack(canvas: np.ndarray, reveal_up_to: Optional[int] = None,
               phase_label: str = "Action Trajectory") -> np.ndarray:
        """Append the action chart underneath ``canvas`` if actions are present."""
        if chart_h <= 0:
            return canvas
        if gt_act_traj is None:
            blank = np.zeros((chart_h, W, 3), dtype=np.uint8)
            return add_divider(canvas, blank, divider_h)
        chart = action_comparison_image(
            pred_act_traj, gt_act_traj,
            width=W, height=chart_h,
            reveal_up_to=reveal_up_to,
            phase_label=phase_label,
        )
        return add_divider(canvas, chart, divider_h)

    def _write(frame_bgr: np.ndarray, copies: int = 1, per_frame_ms: int = 80):
        """Write frame to MP4 (with hold copies) and collect for GIF."""
        for _ in range(max(copies, 1)):
            writer.write(frame_bgr)
        composed_frames.append(frame_bgr.copy())
        frame_durations_ms.append(per_frame_ms)

    # ==================================================================
    # PHASE 1 — Context-frame slideshow (slow, easy to read)
    # ==================================================================
    n_frames = len(frames)
    hold_copies = max(int(frame_hold * fps), 1)
    hold_ms = int(frame_hold * 1000)

    for idx, raw_frame in enumerate(frames):
        sub = f"frame {idx + 1} / {n_frames}"
        panel_lines: List[Tuple[str, Tuple[int, int, int], bool]] = []
        panel_lines.append(("Question", (160, 200, 255), True))
        panel_lines.append((question, (245, 245, 250), False))
        panel_lines.append(("Ground Truth", (255, 210, 120), True))
        panel_lines.append((gt_answer, (250, 230, 180), False))

        canvas = compose_frame(
            raw_frame, phase_label="CONTEXT", sub_label=sub,
            panel_lines=panel_lines, accent=(96, 200, 255),
            active_phase=_PHASE_LABEL_TO_PRAI["CONTEXT"],
        )
        combined = _stack(canvas, reveal_up_to=0,
                          phase_label="Action (GT preview)")
        _write(combined, copies=hold_copies, per_frame_ms=hold_ms)

    # ==================================================================
    # PHASE 2 — Token-by-token prediction reveal
    # ==================================================================
    last_frame = frames[-1].copy()
    if pred_answer is not None:
        pred_words = pred_answer.split() or [""]
        pred_colour = (130, 235, 170) if is_correct else (235, 130, 150)
        token_ms = int(1000 / max(fps, 1)) * 2  # readable typing speed

        for n_words in range(1, len(pred_words) + 1):
            partial = " ".join(pred_words[:n_words])
            cursor = "|" if n_words < len(pred_words) else ""

            panel_lines = []
            panel_lines.append(("Question", (160, 200, 255), True))
            panel_lines.append((question, (245, 245, 250), False))
            panel_lines.append(("Ground Truth", (255, 210, 120), True))
            panel_lines.append((gt_answer, (250, 230, 180), False))
            panel_lines.append(("Prediction", pred_colour, True))
            panel_lines.append((f"{partial}{cursor}", pred_colour, False))

            sub = f"generating ({n_words}/{len(pred_words)})"
            canvas = compose_frame(
                last_frame, phase_label="ANSWER", sub_label=sub,
                panel_lines=panel_lines, is_correct=is_correct,
                accent=(150, 220, 255),
                active_phase=_PHASE_LABEL_TO_PRAI["ANSWER"],
            )
            combined = _stack(canvas, reveal_up_to=0,
                              phase_label="Action (GT preview)")
            _write(combined, copies=max(token_ms * fps // 1000, 1),
                   per_frame_ms=token_ms)

        # Hold the completed prediction briefly so the reader can absorb it.
        _write(combined, copies=max(fps, 1), per_frame_ms=900)

    # ==================================================================
    # PHASE 3 — Animated action trajectory (predicted line drawn step-by-step)
    # ==================================================================
    if has_actions and gt_act_traj is not None:
        n_pred_steps = pred_act_traj.shape[0] if pred_act_traj is not None else 0

        panel_lines = []
        panel_lines.append(("Question", (160, 200, 255), True))
        panel_lines.append((question, (245, 245, 250), False))
        panel_lines.append(("Ground Truth", (255, 210, 120), True))
        panel_lines.append((gt_answer, (250, 230, 180), False))
        if pred_answer is not None:
            pred_colour = (130, 235, 170) if is_correct else (235, 130, 150)
            panel_lines.append(("Prediction", pred_colour, True))
            panel_lines.append((pred_answer, pred_colour, False))

        n_anim = max(n_pred_steps, 1)
        anim_ms = int(1000 / fps)
        for reveal in range(0, n_anim + 1):
            canvas = compose_frame(
                last_frame, phase_label="ACTION",
                sub_label=f"step {reveal} / {n_anim}",
                panel_lines=panel_lines, is_correct=is_correct,
                accent=(255, 180, 110),
                active_phase=_PHASE_LABEL_TO_PRAI["ACTION"],
            )
            combined = _stack(
                canvas,
                reveal_up_to=reveal if pred_act_traj is not None else None,
                phase_label="Action Trajectory",
            )
            _write(combined, copies=1, per_frame_ms=anim_ms)

        # Hold the final chart.
        canvas = compose_frame(
            last_frame, phase_label="ACTION",
            sub_label=f"complete ({n_anim}/{n_anim})",
            panel_lines=panel_lines, is_correct=is_correct,
            accent=(255, 180, 110),
            active_phase=_PHASE_LABEL_TO_PRAI["ACTION"],
        )
        final_combined = _stack(canvas, reveal_up_to=n_anim,
                                phase_label="Action Trajectory")
        _write(final_combined, copies=max(fps * 2, 1), per_frame_ms=1200)

    writer.release()

    if save_gif:
        from PIL import Image as PILImage

        gif_path = output_path.rsplit(".", 1)[0] + ".gif"
        pil_frames = []
        for f in composed_frames:
            rgb = f[:, :, ::-1].copy()
            pil_frames.append(PILImage.fromarray(rgb))

        if pil_frames:
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=frame_durations_ms,
                loop=0,
            )
            logger.info("Saved GIF      -> {}", gif_path)

    logger.info("Saved video    -> {}  ({}x{})", output_path, W, total_H)
