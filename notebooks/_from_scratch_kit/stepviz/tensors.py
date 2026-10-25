"""Pictures of tensors: images, feature-map mosaics, convolution kernels, matrices, histograms.

Every function accepts ``torch.Tensor`` *or* ``numpy.ndarray`` and an optional ``ax``. Without an ``ax``
it draws a standalone figure and shows it inline; with one, it paints into that axes, so the same call
works in a static picture, inside a multi-panel figure and inside an :class:`~stepviz.Animator` frame.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

from ._core import COLORS, MAGNITUDE_CMAP, SIGNED_CMAP, display_figure, new_figure


def as_numpy(t) -> np.ndarray:
    """Detach / move / convert a tensor (or pass an array through) to a float numpy array."""
    if hasattr(t, "detach"):
        t = t.detach().float().cpu().numpy()
    return np.asarray(t, dtype=np.float32)


def _normalise01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _standalone(figsize):
    fig = new_figure(figsize)
    return fig, fig.add_subplot(111)


def image(t, title: Optional[str] = None, ax: Optional[Axes] = None, normalise: bool = True) -> None:
    """Show an image tensor: ``(3, H, W)``, ``(1, 3, H, W)``, ``(H, W, 3)`` or ``(H, W)``."""
    a = as_numpy(t)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    own = ax is None
    if own:
        fig, ax = _standalone((3.2, 3.2))
    a = _normalise01(a) if normalise else np.clip(a, 0, 1)
    ax.imshow(a, cmap=None if a.ndim == 3 else "gray")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    if own:
        display_figure(fig)


def feature_maps(t, n: int = 16, cols: int = 8, title: Optional[str] = None, signed: bool = True,
                 fig=None, show: bool = True) -> None:
    """Mosaic of the first ``n`` channels of a ``(C, H, W)`` or ``(1, C, H, W)`` activation.

    Each channel is a small grey/red-blue picture of *where* that filter fired. ``signed=True`` uses a
    diverging map centred on zero (pre-activation values); ``False`` a magnitude map (post-ReLU).
    """
    a = as_numpy(t)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 2:
        a = a[None]
    n = min(n, a.shape[0])
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    own = fig is None
    if own:
        fig = new_figure((1.5 * cols, 1.5 * rows + (0.5 if title else 0)))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        ch = a[i]
        if signed:
            m = float(np.abs(ch).max()) or 1.0
            ax.imshow(ch, cmap=SIGNED_CMAP, vmin=-m, vmax=m)
        else:
            ax.imshow(ch, cmap=MAGNITUDE_CMAP)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"ch {i}", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    if own and show:
        display_figure(fig)


def kernels(weight, n: int = 16, cols: int = 8, title: Optional[str] = None) -> None:
    """Show convolution kernels ``(C_out, C_in, kh, kw)``.

    With ``C_in == 3`` (a first layer) each kernel is drawn as a tiny RGB image - you literally see the
    colour/edge pattern it looks for. Otherwise the mean over input channels is drawn (red = positive).
    """
    w = as_numpy(weight)
    n = min(n, w.shape[0])
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig = new_figure((1.2 * cols, 1.2 * rows + (0.5 if title else 0)))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        if w.shape[1] == 3:
            ax.imshow(_normalise01(np.transpose(w[i], (1, 2, 0))), interpolation="nearest")
        else:
            k = w[i].mean(0)
            m = float(np.abs(k).max()) or 1.0
            ax.imshow(k, cmap=SIGNED_CMAP, vmin=-m, vmax=m, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    display_figure(fig)


def matrix(m, title: Optional[str] = None, ax: Optional[Axes] = None,
           row_labels: Optional[Sequence[str]] = None, col_labels: Optional[Sequence[str]] = None,
           annotate: Optional[bool] = None, cmap: Optional[str] = None, fmt: str = "{:.2f}",
           highlight_row: Optional[int] = None, highlight_col: Optional[int] = None,
           vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """Heatmap of a 2-D tensor (attention weights, masks, small weight matrices).

    ``annotate`` defaults to writing the numbers when the matrix is small enough to read (≤ 12×12).
    ``highlight_row`` / ``highlight_col`` draw a yellow outline - used by animations that fill a
    matrix one row (query) or one cell at a time.
    """
    a = as_numpy(m)
    if a.ndim == 1:
        a = a[None]
    own = ax is None
    if own:
        size = (min(1.0 + 0.55 * a.shape[1], 10), min(0.8 + 0.5 * a.shape[0], 8))
        fig, ax = _standalone(size)
    signed = a.min() < 0 < a.max()
    if cmap is None:
        cmap = SIGNED_CMAP if signed else "Blues"
    if signed and vmin is None and vmax is None:
        lim = float(np.abs(a).max()) or 1.0
        vmin, vmax = -lim, lim
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if annotate is None:
        annotate = a.shape[0] <= 12 and a.shape[1] <= 12
    if annotate:
        lo = vmin if vmin is not None else float(a.min())
        hi = vmax if vmax is not None else float(a.max())
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                v = a[r, c]
                if np.isnan(v):
                    continue
                strong = abs(v - (lo + hi) / 2) > 0.3 * (hi - lo) if hi > lo else False
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if strong else COLORS["text"])
    ax.set_xticks(range(a.shape[1]))
    ax.set_yticks(range(a.shape[0]))
    ax.set_xticklabels(col_labels if col_labels is not None else range(a.shape[1]), fontsize=8,
                       rotation=45 if col_labels is not None else 0)
    ax.set_yticklabels(row_labels if row_labels is not None else range(a.shape[0]), fontsize=8)
    from matplotlib.patches import Rectangle

    if highlight_row is not None:
        ax.add_patch(Rectangle((-0.5, highlight_row - 0.5), a.shape[1], 1, fill=False,
                               edgecolor=COLORS["active"], lw=3))
    if highlight_col is not None:
        ax.add_patch(Rectangle((highlight_col - 0.5, -0.5), 1, a.shape[0], fill=False,
                               edgecolor=COLORS["compare"], lw=3))
    if title:
        ax.set_title(title, fontsize=10)
    if own:
        display_figure(fig)


def histogram(tensors: Dict[str, object], title: str = "value distribution", bins: int = 60) -> None:
    """Overlaid histograms - e.g. activations before vs after a normalisation layer."""
    fig, ax = _standalone((7, 3))
    for name, t in tensors.items():
        ax.hist(as_numpy(t).ravel(), bins=bins, alpha=0.55, label=name, density=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    display_figure(fig)


def panels(items: Dict[str, object], kind: str = "image", cols: int = 4,
           title: Optional[str] = None) -> None:
    """A row/grid of labelled pictures (``kind`` = ``"image"`` or ``"matrix"`` or ``"map"``).

    ``"map"`` shows the channel-mean of a feature map - a quick "where is the energy" view.
    """
    names = list(items)
    cols = min(cols, len(names))
    rows = math.ceil(len(names) / cols)
    fig = new_figure((3.0 * cols, 2.8 * rows + (0.4 if title else 0)))
    for i, name in enumerate(names):
        ax = fig.add_subplot(rows, cols, i + 1)
        t = items[name]
        if kind == "image":
            image(t, title=name, ax=ax)
        elif kind == "matrix":
            matrix(t, title=name, ax=ax, annotate=False)
        elif kind == "map":
            a = as_numpy(t)
            while a.ndim > 2:
                a = a.mean(0) if a.ndim == 3 else a[0]
            ax.imshow(a, cmap=MAGNITUDE_CMAP)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=9)
        else:
            raise ValueError(f"unknown kind {kind!r}")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    display_figure(fig)


def curve(series: Dict[str, Sequence[float]], title: str, xlabel: str = "step",
          ylabel: str = "value", logy: bool = False) -> None:
    """Line plot of one or more series (loss curves, schedules, metric sweeps)."""
    fig, ax = _standalone((7, 3.2))
    for name, ys in series.items():
        ax.plot(list(ys), lw=2, label=name)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    display_figure(fig)
