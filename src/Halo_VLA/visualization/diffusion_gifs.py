"""Animated GIFs of the visual DiT: denoising snapshots, filmstrips and autoregressive rollouts."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from visualization.action_plots import (  # noqa: F401 - moved; re-exported for callers
    action_comparison_image,
    tensor_to_display_image,
    unnormalise_image,
)
from visualization.autoregressive_gif import (  # noqa: F401 - moved; re-exported for callers
    _AR_ANCHOR_MS,
    _AR_DENOISE_SNAPSHOTS,
    _filmstrip_bar,
    _heun_with_snapshots,
    generate_autoregressive_gif,
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

# Map the phase_label used in compose_frame → PRAI index.
_PHASE_LABEL_TO_PRAI = {
    "CONTEXT":       0,
    "ANSWER":        1,
    "ACTION":        2,
    "NOISE -> IMAGE": 3,
}


# ---------------------------------------------------------------------------
# DiT diffusion intermediates: re-run the Euler integration but capture the
# intermediate ``x`` after each step so we can render a noise -> image GIF.
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_dit_with_intermediates(
    predictor,
    text_emb: torch.Tensor,
    height: int,
    width: int,
    num_frames: int = 1,
    num_steps: int = 30,
    num_snapshots: int = 24,
    world_video_query_hiddens: Optional[torch.Tensor] = None,
):
    """
    Mirror of ``RGBFramePredictor.predict_future_frames`` that records
    intermediate ``x`` values during Euler integration.

    Returns:
        snapshots: list of [B, num_frames, C, H, W] tensors (CPU), one per
                   captured step (always includes the initial noise and the
                   final sample).
        step_times: list[float] giving the t in [0, 1] each snapshot represents.
    """
    num_frames = min(num_frames, predictor.num_predict_frames)
    B = text_emb.size(0)
    rgb_channels = getattr(predictor.config, "rgb_channels", 3)

    cond_all = predictor._create_all_frame_contexts(
        text_emb, num_frames, world_video_query_hiddens,
    )  # [B*F, D]

    n = cond_all.size(0)
    device = cond_all.device
    x = torch.randn(n, rgb_channels, height, width, device=device)
    dt = 1.0 / num_steps

    # Choose which Euler steps to snapshot. Always include the start (noise)
    # and the final step, and spread the rest evenly across [0, num_steps].
    if num_snapshots >= num_steps + 1:
        keep = set(range(num_steps + 1))
    else:
        idxs = np.linspace(0, num_steps, num_snapshots).round().astype(int)
        keep = set(int(i) for i in idxs)

    snapshots: List[torch.Tensor] = []
    step_times: List[float] = []

    def _record(step_idx: int, x_now: torch.Tensor):
        snap = x_now.view(B, num_frames, rgb_channels, height, width).cpu()
        snapshots.append(snap)
        step_times.append(step_idx * dt)

    if 0 in keep:
        _record(0, x)

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((n,), t_val, device=device)
        v = predictor.dit_rgb(x, t, cond_all)
        x = x + v * dt
        if (i + 1) in keep:
            _record(i + 1, x)

    return snapshots, step_times


def save_diffusion_gif(
    snapshots: List[torch.Tensor],
    step_times: List[float],
    output_path: str,
    frame_idx: int = 0,
    sample_idx: int = 0,
    canvas_w: int = 384,
    canvas_h: int = 384,
    fps: int = 12,
    hold_final_seconds: float = 1.5,
    img_mean: Tuple = (0.485, 0.456, 0.406),
    img_std: Tuple = (0.229, 0.224, 0.225),
):
    """
    Render the noise -> image diffusion trajectory for a single predicted
    frame as an animated GIF. Each GIF frame is annotated with the step
    progress and the diffusion time t.

    img_mean / img_std must match the normalization used during training so
    that the final snapshot (t≈1, which lives in ImageNet-normalized space)
    is correctly converted to display colors.  Intermediate steps use
    per-channel min-max normalization because they are a blend of noise and
    normalized data that doesn't fit a fixed range.
    """
    cv2 = _import_cv2()
    from PIL import Image as PILImage

    if not snapshots:
        return

    n_steps = len(snapshots)
    last_idx = n_steps - 1
    pil_frames = []
    durations = []
    base_dur = int(1000 / fps)

    for i, (snap, t_val) in enumerate(zip(snapshots, step_times)):
        # snap: [B, F, C, H, W]
        img_t = snap[sample_idx, frame_idx]  # [C, H, W]

        # Final snapshot is a proper ImageNet-normalized frame — use the same
        # unnormalization as the main video so colors are accurate.  All
        # intermediate snapshots are noise/frame blends; min-max norm suffices.
        if i == last_idx:
            img = unnormalise_image(img_t, img_mean, img_std)
        else:
            img = tensor_to_display_image(img_t)
        img = cv2.resize(img, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)

        # Bug-fix: first snapshot is step 0 (pure noise) → progress = 0 %
        progress = i / max(last_idx, 1)
        sub = f"step {i}/{last_idx}    t = {t_val:.2f}"
        panel_lines = [
            ("DiT velocity-field denoising", (200, 215, 255), False),
            (f"Progress: {int(progress * 100):3d}%", (255, 255, 255), True),
        ]
        # Header colour shifts from cool (noise) to warm (data) so the user
        # can feel the progress at a glance.
        accent = (
            int(96 + (255 - 96) * progress),
            int(200 - (200 - 170) * progress),
            int(255 - (255 - 100) * progress),
        )
        composed = compose_frame(
            img, phase_label="NOISE -> IMAGE",
            sub_label=sub, panel_lines=panel_lines,
            accent=accent,
            active_phase=_PHASE_LABEL_TO_PRAI["NOISE -> IMAGE"],
        )

        rgb = composed[:, :, ::-1].copy()
        pil_frames.append(PILImage.fromarray(rgb))
        durations.append(base_dur)

    # Hold the final frame for an extra moment so the result is readable.
    if pil_frames:
        hold = int(max(hold_final_seconds, 0) * 1000)
        if hold > 0:
            durations[-1] = durations[-1] + hold

        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
        )
        logger.info("Saved diffusion GIF -> {}", output_path)










