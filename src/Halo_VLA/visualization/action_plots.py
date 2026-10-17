"""Action trajectories and image tensors rendered for display."""

from typing import Optional, Tuple

import numpy as np
import torch

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


# ---------------------------------------------------------------------------
# Action comparison plot — polished dark theme with distinct per-DoF colours,
# legend in two columns underneath the axes (so it never occludes data), a
# vertical cursor at the latest revealed timestep, and a clear ASCII title.
# ---------------------------------------------------------------------------
def action_comparison_image(pred: Optional[np.ndarray],
                            gt: np.ndarray,
                            width: int = 640,
                            height: int = 240,
                            reveal_up_to: Optional[int] = None,
                            phase_label: str = "Action Trajectory") -> np.ndarray:
    """
    Render predicted vs GT action trajectories.

    Solid colour-coded line per DoF for GT, dashed line of the same colour
    family for the prediction, plus a vertical cursor at the current step.
    Each dimension uses a distinct hue from the ``tab20`` palette.
    """
    plt = _import_matplotlib()
    import matplotlib.cm as cm

    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    if pred is not None and pred.ndim == 1:
        pred = pred[np.newaxis, :]

    n_gt_steps, n_dims = gt.shape
    gt_t = np.arange(n_gt_steps)

    # Palette: tab20 gives 20 distinct hues. We use even indices for GT and
    # the matching odd index for the predicted version of the same DoF, so
    # GT-d0 and Pred-d0 share a colour family but are visually separable.
    base = cm.get_cmap("tab20", 20)
    gt_colours = [base(min((d * 2) % 20, 19)) for d in range(n_dims)]
    pred_colours = [base(min((d * 2 + 1) % 20, 19)) for d in range(n_dims)]

    with plt.rc_context({
        "figure.facecolor": "#0d1226",
        "axes.facecolor": "#141a30",
        "axes.edgecolor": "#3a4060",
        "axes.labelcolor": "#dde2f0",
        "xtick.color": "#a0a8c0",
        "ytick.color": "#a0a8c0",
        "text.color": "#e8ecf7",
        "grid.color": "#242a45",
        "grid.linestyle": "--",
        "grid.alpha": 0.55,
        "axes.titleweight": "bold",
    }):
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # GT lines — solid, medium weight.
        for d in range(n_dims):
            colour = gt_colours[d]
            ax.plot(gt_t, gt[:, d], color=colour, linewidth=1.6,
                    alpha=0.95,
                    label=f"GT d{d}" if d < 8 else "_nolegend_")

        # Predicted lines — dashed, with a cursor marker at the latest
        # revealed step so the user can see the animation progressing.
        if pred is not None:
            n_pred_steps = pred.shape[0]
            pred_t = np.arange(n_pred_steps)
            k = min(reveal_up_to, n_pred_steps) if reveal_up_to is not None else n_pred_steps

            if k > 0:
                pred_slice = pred[:k, :]
                pred_t_slice = pred_t[:k]

                for d in range(min(n_dims, pred_slice.shape[1])):
                    colour = pred_colours[d]
                    ax.plot(pred_t_slice, pred_slice[:, d],
                            color=colour, linewidth=1.7, alpha=0.95,
                            linestyle=(0, (4, 2)),
                            label=f"Pred d{d}" if d < 8 else "_nolegend_")

                    if k < n_pred_steps:
                        ax.plot(pred_t_slice[-1], pred_slice[-1, d], "o",
                                color=colour, markersize=5, zorder=6,
                                markeredgecolor="#ffffff", markeredgewidth=0.8)

                # Vertical cursor at the currently revealed timestep.
                if k < n_pred_steps and k >= 1:
                    ax.axvline(x=pred_t_slice[-1], color="#ffffff",
                               linewidth=0.6, alpha=0.25, linestyle=":")

        # Subtle horizontal zero line for reference.
        ax.axhline(y=0, color="#5a6488", linewidth=0.6, alpha=0.5,
                   linestyle="-")

        # Legend below the plot, two rows, so it never overlaps the data.
        ax.legend(
            fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.42),
            facecolor="#0d1226", edgecolor="#3a4060",
            labelcolor="#dde2f0", framealpha=0.95,
            ncol=8, borderaxespad=0.2, handlelength=2.2, columnspacing=1.0,
        )

        ax.set_xlabel("Timestep", fontsize=8.5)
        ax.set_ylabel("Value", fontsize=8.5)
        ax.set_title(f"{phase_label}   GT (solid)  /  Pred (dashed)",
                     fontsize=9.5, color="#ffffff", pad=6)
        ax.tick_params(labelsize=7.5, length=3, width=0.7)
        ax.grid(True, linewidth=0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        fig.tight_layout(pad=0.6, rect=(0, 0.08, 1, 1))

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
        buf = buf.reshape((h, w, 4))[:, :, :3]
        plt.close(fig)

    cv2 = _import_cv2()
    buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    buf = cv2.resize(buf, (width, height))
    return buf


# ---------------------------------------------------------------------------
# Image normalisation helpers
# ---------------------------------------------------------------------------
def unnormalise_image(tensor: torch.Tensor, mean: Tuple, std: Tuple) -> np.ndarray:
    """Convert a normalised [3, H, W] tensor to uint8 BGR for OpenCV."""
    cv2 = _import_cv2()
    img = tensor.clone().cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def tensor_to_display_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert an arbitrary [3, H, W] float tensor (possibly raw DiT latent) to a
    displayable BGR uint8 image by per-channel min-max normalisation.

    Used for visualising noisy intermediate diffusion steps where the values
    don't fit the ImageNet (mean, std) re-normalisation range.
    """
    cv2 = _import_cv2()
    t = t.detach().cpu().float()
    t_min = t.amin(dim=(1, 2), keepdim=True)
    t_max = t.amax(dim=(1, 2), keepdim=True)
    t = (t - t_min) / (t_max - t_min + 1e-6)
    img = t.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
