"""Step-by-step animations of the primitive operations every architecture is built from.

Each function runs the operation *by hand* on a small example (so every number on screen is real),
records one frame per elementary step and embeds the GIF. They also return the computed result, so a
notebook can ``assert`` it equals the PyTorch op - the picture is guaranteed to match the maths.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from ._core import COLORS, Animator, caption, heading
from .tensors import as_numpy


def _grid(ax: Axes, a: np.ndarray, title: str = "", window=None, window_color: str = "active",
          cell_colors: Optional[dict] = None, fmt: str = "{:g}", empty=None, muted=None) -> None:
    """Draw a small 2-D array as a table of numbers.

    ``window`` = (r0, c0, h, w) outlined region; ``cell_colors`` = {(r, c): role};
    ``empty`` = boolean mask of cells not computed yet; ``muted`` = boolean mask of padding cells.
    """
    rows, cols = a.shape
    for r in range(rows):
        for c in range(cols):
            face = "white"
            if muted is not None and muted[r, c]:
                face = COLORS["muted"]
            if cell_colors and (r, c) in cell_colors:
                face = COLORS.get(cell_colors[(r, c)], cell_colors[(r, c)])
            ax.add_patch(Rectangle((c, rows - 1 - r), 1, 1, facecolor=face,
                                   edgecolor=COLORS["edge"], lw=1))
            if empty is None or not empty[r, c]:
                ax.text(c + 0.5, rows - 0.5 - r, fmt.format(round(float(a[r, c]), 2)),
                        ha="center", va="center", fontsize=9 if max(rows, cols) <= 8 else 7)
    if window is not None:
        r0, c0, h, w = window
        ax.add_patch(Rectangle((c0, rows - r0 - h), w, h, fill=False,
                               edgecolor=COLORS.get(window_color, window_color), lw=3.5))
    ax.set_xlim(-0.2, cols + 0.2)
    ax.set_ylim(-0.2, rows + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")


def conv2d_animation(x, k, stride: int = 1, padding: int = 0, max_frames: int = 12,
                     title: str = "2-D convolution, one output pixel per frame") -> np.ndarray:
    """Animate a single-channel convolution (really cross-correlation, like ``nn.Conv2d``).

    Frame *i*: the kernel window over the (padded) input, the element-wise products summed into one
    number, and that number appearing in the output grid. Returns the full output array.
    """
    x = as_numpy(x)
    k = as_numpy(k)
    kh, kw = k.shape
    xp = np.pad(x, padding)
    pad_mask = np.ones_like(xp, dtype=bool)
    pad_mask[padding:padding + x.shape[0], padding:padding + x.shape[1]] = False
    oh = (xp.shape[0] - kh) // stride + 1
    ow = (xp.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    positions = [(r, c) for r in range(oh) for c in range(ow)]
    for r, c in positions:
        out[r, c] = float((xp[r * stride:r * stride + kh, c * stride:c * stride + kw] * k).sum())
    # Show the first frames step by step, then jump to the end so long examples stay short.
    shown = positions[:max_frames - 1] + ([positions[-1]] if len(positions) >= max_frames else [])
    anim = Animator(figsize=(11, 4.2))
    for idx, (r, c) in enumerate(shown):
        done = np.ones((oh, ow), dtype=bool)
        for rr, cc in positions[:positions.index((r, c)) + 1]:
            done[rr, cc] = False
        window = xp[r * stride:r * stride + kh, c * stride:c * stride + kw]
        terms = " + ".join(f"{a:g}·{b:g}" for a, b in zip(window.ravel()[:4], k.ravel()[:4]))
        more = " + …" if kh * kw > 4 else ""
        with anim.frame() as fig:
            heading(fig, title)
            ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [xp.shape[1], kw, ow]})
            _grid(ax1, xp, f"input (padding={padding})", window=(r * stride, c * stride, kh, kw),
                  muted=pad_mask)
            _grid(ax2, k, "kernel", cell_colors={(i, j): "compare" for i in range(kh)
                                                  for j in range(kw)})
            _grid(ax3, out, f"output {oh}×{ow}", empty=done,
                  cell_colors={(r, c): "done"})
            caption(fig, f"out[{r},{c}] = {terms}{more} = {out[r, c]:g}"
                    + ("   (… jumping to the last pixel)" if idx == len(shown) - 1
                       and len(positions) > len(shown) else ""))
    anim.show()
    return out


def pool_animation(x, size: int = 2, stride: Optional[int] = None, mode: str = "max",
                   title: Optional[str] = None) -> np.ndarray:
    """Animate max- or average-pooling: each frame picks one window and writes one output cell."""
    x = as_numpy(x)
    stride = stride or size
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    anim = Animator(figsize=(9, 4))
    filled = np.ones((oh, ow), dtype=bool)
    for r in range(oh):
        for c in range(ow):
            win = x[r * stride:r * stride + size, c * stride:c * stride + size]
            out[r, c] = win.max() if mode == "max" else win.mean()
            filled[r, c] = False
            colors = {}
            if mode == "max":
                wr, wc = np.unravel_index(int(win.argmax()), win.shape)
                colors[(r * stride + wr, c * stride + wc)] = "done"
            with anim.frame() as fig:
                heading(fig, title or f"{mode}-pooling {size}×{size}, stride {stride}")
                ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [x.shape[1], ow]})
                _grid(ax1, x, "input", window=(r * stride, c * stride, size, size),
                      cell_colors=colors)
                _grid(ax2, out, f"output {oh}×{ow}", empty=filled.copy(),
                      cell_colors={(r, c): "done"})
                what = "largest value" if mode == "max" else "mean"
                caption(fig, f"window ({r},{c}) -> {what} = {out[r, c]:g}")
    anim.show()
    return out


def patchify_animation(img, patch: int, max_frames: int = 16,
                       title: str = "Cutting an image into patch tokens") -> np.ndarray:
    """Animate ViT-style patchification: each frame cuts out one patch and appends it as a token.

    Returns the patches as ``(N, patch*patch*C)`` rows - the exact matrix a patch embedding multiplies.
    """
    a = as_numpy(img)
    if a.ndim == 4:
        a = a[0]
    c, h, w = a.shape
    gh, gw = h // patch, w // patch
    rgb = np.transpose(a, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    patches = [rgb[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch]
               for i in range(gh) for j in range(gw)]
    n = len(patches)
    order = list(range(min(n, max_frames - 1))) + ([n - 1] if n >= max_frames else [])
    anim = Animator(figsize=(11, 4.4))
    for k in order:
        i, j = divmod(k, gw)
        with anim.frame() as fig:
            heading(fig, title)
            gs = fig.add_gridspec(2, max(n, 1) if n <= 16 else 16, height_ratios=[3, 1])
            ax = fig.add_subplot(gs[0, :])
            ax.imshow(rgb, extent=(0, w, h, 0))
            for t in range(1, gw):
                ax.axvline(t * patch, color="white", lw=1)
            for t in range(1, gh):
                ax.axhline(t * patch, color="white", lw=1)
            ax.add_patch(Rectangle((j * patch, i * patch), patch, patch, fill=False,
                                   edgecolor=COLORS["active"], lw=3))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            strip = min(n, 16)
            for s in range(strip):
                tok = s if n <= 16 else int(round(s * (n - 1) / 15))
                tax = fig.add_subplot(gs[1, s])
                if tok <= k:
                    tax.imshow(patches[tok])
                tax.set_xticks([])
                tax.set_yticks([])
                tax.set_title(str(tok), fontsize=7)
            caption(fig, f"patch ({i},{j}) -> token {k} of {n}   "
                         f"(each token = {patch}×{patch}×{c} = {patch * patch * c} numbers)")
    anim.show()
    flat = a.reshape(c, gh, patch, gw, patch).transpose(1, 3, 2, 4, 0).reshape(n, -1)
    return flat


def attention_animation(q, k, labels: Optional[Sequence[str]] = None, mask=None,
                        title: str = "Attention: one query row per frame") -> np.ndarray:
    """Animate scaled dot-product attention weights, one query (row) at a time.

    Frame *i*: raw scores ``q_i · k_j / sqrt(d)`` for every key, then the softmax turning that row
    into weights that sum to 1. ``mask`` uses the HaleBlocks convention: ``True`` = blocked.
    Returns the full ``(n_q, n_k)`` weight matrix.
    """
    q = as_numpy(q)
    k = as_numpy(k)
    d = q.shape[-1]
    scores = q @ k.T / math.sqrt(d)
    if mask is not None:
        scores = np.where(as_numpy(mask).astype(bool), -np.inf, scores)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)
    labels = list(labels) if labels is not None else [str(i) for i in range(q.shape[0])]
    klabels = labels if k.shape[0] == q.shape[0] else [str(j) for j in range(k.shape[0])]
    shown_scores = np.where(np.isinf(scores), np.nan, scores)
    anim = Animator(figsize=(11, 4.6))
    from .tensors import matrix

    for i in range(q.shape[0]):
        partial = np.full_like(weights, np.nan)
        partial[:i + 1] = weights[:i + 1]
        with anim.frame() as fig:
            heading(fig, title)
            ax1, ax2 = fig.subplots(1, 2)
            matrix(shown_scores, ax=ax1, title="scores = q·kᵀ/√d  (grey = masked)",
                   row_labels=labels, col_labels=klabels, highlight_row=i)
            matrix(partial, ax=ax2, title="softmax(scores) row by row", row_labels=labels,
                   col_labels=klabels, highlight_row=i, cmap="Blues", vmin=0, vmax=1)
            best = int(np.nanargmax(weights[i]))
            caption(fig, f"query '{labels[i]}' attends most to '{klabels[best]}' "
                         f"(weight {weights[i, best]:.2f}); every row sums to 1")
    anim.show()
    return weights
