"""Shared plumbing: colour roles, inline display and the frame-by-frame GIF animator.

Design decisions
----------------
* Every frame is drawn on a detached :class:`matplotlib.figure.Figure` (never ``pyplot``), so building
  a 40-frame animation does not spam 40 inline plots and never touches the notebook's backend.
* GIFs are embedded straight into the notebook output (``IPython.display.Image(data=...)``) instead of
  being written next to it. The from-scratch notebooks are copied into several repositories; keeping
  the animations inside the ``.ipynb`` means a copy is always complete and no asset folder can go stale.
* :meth:`Animator.frame` is a context manager: "open a fresh figure, draw whatever this step needs,
  close it". Heterogeneous steps (a matrix here, an image there) therefore need no common schema.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Colour roles shared with the DSA roadmap notebooks: the same colour always means the same thing.
COLORS = {
    "input": "#F5B041",  # data entering a step
    "active": "#F7C548",  # what the step is looking at right now
    "compare": "#7FB3E6",  # the other operand (kernel, key, ...)
    "done": "#6FCF97",  # finished / output
    "bad": "#EB7A7A",  # rejected / masked
    "muted": "#D6DBDF",  # ignored
    "accent": "#B98ADB",  # special marker (class token, residual path ...)
    "edge": "#2C3E50",
    "text": "#1B2631",
    "paper": "#FFFFFF",
}

# Diverging colour map for signed activations / weights, sequential one for magnitudes.
SIGNED_CMAP = "RdBu_r"
MAGNITUDE_CMAP = "viridis"


def new_figure(figsize: Tuple[float, float] = (8, 4), dpi: int = 100) -> Figure:
    """A detached figure with a canvas attached (ready for ``savefig`` / pixel export)."""
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=COLORS["paper"])
    FigureCanvasAgg(fig)
    return fig


def display_figure(fig: Figure) -> None:
    """Show a detached figure inline as a PNG (works in Jupyter, VS Code and nbclient builds)."""
    from IPython.display import Image, display  # lazy: the kit stays importable outside IPython

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    display(Image(data=buf.getvalue(), format="png"))


class Animator:
    """Collect frames and embed them in the notebook as a looping GIF.

    Usage::

        anim = Animator(figsize=(8, 3.5))
        for step in steps:
            with anim.frame() as fig:
                ax = fig.add_subplot()
                ...draw this step...
        anim.show()                       # embeds the GIF in the cell output

    Parameters
    ----------
    figsize, dpi:
        Fixed for every frame (a GIF needs a constant pixel size).
    ms_per_frame:
        Display time of each frame; the last one lingers for ``hold_last_ms``.
    """

    def __init__(self, figsize: Tuple[float, float] = (8, 3.6), dpi: int = 80,
                 ms_per_frame: int = 900, hold_last_ms: int = 2500) -> None:
        self.figsize = figsize
        self.dpi = dpi
        self.ms_per_frame = ms_per_frame
        self.hold_last_ms = hold_last_ms
        self._frames: List["object"] = []

    @contextmanager
    def frame(self) -> Iterator[Figure]:
        """Yield a blank figure; when the block ends it becomes the next frame."""
        from PIL import Image  # lazy: Pillow is only needed for GIFs

        fig = new_figure(self.figsize, self.dpi)
        yield fig
        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        image = Image.frombuffer("RGBA", (width, height), canvas.buffer_rgba()).convert("RGB")
        self._frames.append(image)

    def __len__(self) -> int:
        return len(self._frames)

    def to_bytes(self) -> bytes:
        """Encode all frames as an optimised GIF."""
        if not self._frames:
            raise ValueError("Animator has no frames - draw at least one with `with anim.frame()`.")
        # One adaptive palette per frame keeps colours faithful while staying small enough for git.
        frames = [im.quantize(colors=128, method=2) for im in self._frames]
        durations = [self.ms_per_frame] * (len(frames) - 1) + [self.hold_last_ms]
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True, disposal=2)
        return buf.getvalue()

    def show(self, save_to: Optional[Union[str, Path]] = None) -> None:
        """Embed the GIF in the notebook output (and optionally also write it to ``save_to``)."""
        from IPython.display import Image, display

        data = self.to_bytes()
        if save_to is not None:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            Path(save_to).write_bytes(data)
        display(Image(data=data, format="gif"))


def caption(fig: Figure, text: str, y: float = 0.02, size: int = 11) -> None:
    """One explanatory sentence at the bottom of a figure (the 'what just happened' line)."""
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=size, color=COLORS["text"], wrap=True)


def heading(fig: Figure, text: str, size: int = 13) -> None:
    """Bold title at the top of a figure."""
    fig.suptitle(text, fontsize=size, fontweight="bold", color=COLORS["text"])
