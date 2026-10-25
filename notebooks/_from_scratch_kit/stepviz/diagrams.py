"""Static explanatory figures: the "tensor journey" of a model and labelled block diagrams.

These complement the Mermaid diagrams in the markdown cells: Mermaid shows *structure* (what connects
to what), these show *size* - each tensor is a box whose height grows with the channel count and whose
width grows with the spatial size, so a beginner can see a CNN trade resolution for depth.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ._core import COLORS, display_figure, new_figure

Stage = Tuple[str, Tuple[int, ...]]


def _fmt(shape: Tuple[int, ...]) -> str:
    return "×".join(str(s) for s in shape)


def tensor_journey(stages: Sequence[Stage], title: str = "The tensor's journey",
                   note: Optional[str] = None) -> None:
    """Draw a left-to-right pipeline of tensors.

    ``stages`` = ``[("image", (3, 224, 224)), ("stem", (64, 56, 56)), ...]`` - shapes without the
    batch dimension. 3-D shapes are drawn as (channels, H, W) slabs, 2-D as (tokens, dim) bars and
    1-D as vectors.
    """
    n = len(stages)
    fig = new_figure((max(2.2 * n, 8), 4.2))
    ax = fig.add_subplot(111)
    max_c = max(s[1][0] for s in stages if len(s[1]) >= 1)
    max_hw = max((s[1][-1] for s in stages if len(s[1]) == 3), default=1)
    x = 0.0
    centres = []
    for label, shape in stages:
        if len(shape) == 3:
            c, _, w = shape
            height = 0.6 + 2.4 * math.log2(c + 1) / math.log2(max_c + 1)
            width = 0.25 + 1.2 * w / max_hw
            role = "compare"
        elif len(shape) == 2:
            height = 0.6 + 2.4 * math.log2(shape[0] + 1) / math.log2(max(max_c, shape[0]) + 1)
            width = 0.35
            role = "accent"
        else:
            height, width, role = 0.3, 1.0, "done"
        ax.add_patch(FancyBboxPatch((x, -height / 2), width, height, boxstyle="round,pad=0.03",
                                    facecolor=COLORS[role], edgecolor=COLORS["edge"], lw=1.4))
        ax.text(x + width / 2, height / 2 + 0.25, label, ha="center", va="bottom", fontsize=9,
                fontweight="bold")
        ax.text(x + width / 2, -height / 2 - 0.2, _fmt(shape), ha="center", va="top", fontsize=8,
                color=COLORS["text"])
        centres.append((x, x + width))
        x += width + 1.0
    for (_, right), (left, _) in zip(centres, centres[1:]):
        ax.add_patch(FancyArrowPatch((right + 0.08, 0), (left - 0.08, 0), arrowstyle="-|>",
                                     mutation_scale=14, color=COLORS["edge"]))
    ax.set_xlim(-0.4, x - 0.6)
    ax.set_ylim(-2.4, 2.6)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=9, color=COLORS["text"])
    display_figure(fig)


def block_diagram(boxes: Sequence[Tuple[str, str]], skip: Optional[Tuple[int, int]] = None,
                  title: str = "", vertical: bool = False) -> None:
    """A row of labelled operation boxes with arrows, optionally with a skip connection arc.

    ``boxes`` = ``[("conv 3×3", "compare"), ("BN", "muted"), ("ReLU", "done"), ...]`` (label, role).
    ``skip`` = ``(i, j)`` draws a curved arrow from box ``i``'s input to box ``j``'s output - the
    residual path of a ResNet block, a DenseNet concat, a transformer's ``x + f(x)`` ...
    """
    n = len(boxes)
    fig = new_figure((max(1.7 * n, 6), 2.6) if not vertical else (3.2, max(1.0 * n, 4)))
    ax = fig.add_subplot(111)
    step = 1.6
    for i, (label, role) in enumerate(boxes):
        x, y = (i * step, 0) if not vertical else (0, -i * step * 0.7)
        ax.add_patch(FancyBboxPatch((x, y - 0.3), 1.2, 0.6, boxstyle="round,pad=0.04",
                                    facecolor=COLORS.get(role, role), edgecolor=COLORS["edge"]))
        ax.text(x + 0.6, y, label, ha="center", va="center", fontsize=9, fontweight="bold")
        if i:
            if not vertical:
                ax.add_patch(FancyArrowPatch(((i - 1) * step + 1.25, 0), (x - 0.05, 0),
                                             arrowstyle="-|>", mutation_scale=12,
                                             color=COLORS["edge"]))
            else:
                ax.add_patch(FancyArrowPatch((0.6, y + step * 0.7 - 0.35), (0.6, y + 0.35),
                                             arrowstyle="-|>", mutation_scale=12,
                                             color=COLORS["edge"]))
    if skip is not None and not vertical:
        i, j = skip
        ax.add_patch(FancyArrowPatch((i * step - 0.1, 0.35), (j * step + 1.3, 0.35),
                                     arrowstyle="-|>", mutation_scale=14, color="#8E44AD", lw=2,
                                     connectionstyle="arc3,rad=-0.35"))
        ax.text((i * step + j * step + 1.2) / 2, 1.05, "skip / residual path", ha="center",
                fontsize=8, color="#8E44AD", fontweight="bold")
    if not vertical:
        ax.set_xlim(-0.5, n * step)
        ax.set_ylim(-0.8, 1.4)
    else:
        ax.set_xlim(-0.3, 1.5)
        ax.set_ylim(-n * step * 0.7, 0.7)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    display_figure(fig)
