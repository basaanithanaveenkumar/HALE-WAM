"""Record what every layer of a model does to the data, then show it as a table, gallery or film.

``StepRecorder`` attaches forward hooks to chosen sub-modules, runs one forward pass and keeps each
module's output (detached, on CPU). Because it works on *any* ``nn.Module``, every from-scratch
notebook gets "a picture after each step" of its full model for free, and the pictures are always of
the code the notebook just defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
from matplotlib.patches import Rectangle

from ._core import COLORS, MAGNITUDE_CMAP, Animator, caption, display_figure, heading, new_figure
from .tensors import as_numpy


@dataclass
class Step:
    """One recorded module call."""

    index: int
    name: str  # dotted path inside the model, e.g. "stages.0.1.conv2"
    kind: str  # class name, e.g. "Conv2d"
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    output: np.ndarray  # first batch element only (keeps memory small)
    params: int


def _first_tensor(x):
    """Pick the tensor out of a module input/output (tuples, dataclasses, dicts...)."""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for item in x.values():
            t = _first_tensor(item)
            if t is not None:
                return t
    for attr in ("last_hidden_state", "pooled_output", "logits"):
        if hasattr(x, attr):
            return _first_tensor(getattr(x, attr))
    return None


class StepRecorder:
    """Capture the output of selected sub-modules during one forward pass.

    Parameters
    ----------
    model:
        Any ``nn.Module``.
    kinds:
        Record modules of these classes (e.g. ``(nn.Conv2d, nn.MaxPool2d)``).
    names:
        Or record exactly these dotted names (order of execution is kept either way).
    max_depth:
        Or record every module at most this deep in the module tree (1 = direct children).

    Usage::

        rec = StepRecorder(model, max_depth=1)
        with rec:
            model(x)
        rec.table(); rec.gallery(); rec.film()
    """

    def __init__(self, model, kinds: Optional[Sequence[Type]] = None,
                 names: Optional[Iterable[str]] = None, max_depth: Optional[int] = None,
                 where: Optional[Callable[[str, object], bool]] = None) -> None:
        self.model = model
        self.steps: List[Step] = []
        self._handles = []
        wanted = set(names) if names is not None else None
        for name, module in model.named_modules():
            if not name:
                continue
            depth = name.count(".") + 1
            pick = False
            if kinds is not None and isinstance(module, tuple(kinds)):
                pick = True
            if wanted is not None and name in wanted:
                pick = True
            if max_depth is not None and depth <= max_depth:
                pick = True
            if where is not None and where(name, module):
                pick = True
            if pick:
                self._handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(module, inputs, output):
            t_in, t_out = _first_tensor(inputs), _first_tensor(output)
            if t_out is None:
                return
            self.steps.append(Step(
                index=len(self.steps),
                name=name,
                kind=type(module).__name__,
                in_shape=tuple(t_in.shape) if t_in is not None else (),
                out_shape=tuple(t_out.shape),
                output=as_numpy(t_out[0] if t_out.dim() > 1 else t_out),
                params=sum(p.numel() for p in module.parameters(recurse=True)),
            ))
        return hook

    def __enter__(self) -> "StepRecorder":
        self.steps.clear()
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        """Detach all hooks (also called automatically when the ``with`` block ends)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ views
    def table(self) -> None:
        """Print one line per step: name, type, input shape -> output shape, parameters."""
        print(f"{'#':>3}  {'module':34s} {'type':18s} {'input':>22s} -> {'output':22s} {'params':>10s}")
        for s in self.steps:
            print(f"{s.index:>3}  {s.name[:34]:34s} {s.kind[:18]:18s} {str(s.in_shape):>22s} -> "
                  f"{str(s.out_shape):22s} {s.params:>10,}")

    @staticmethod
    def _summary_image(a: np.ndarray) -> Tuple[np.ndarray, str]:
        """Reduce any activation to a 2-D picture plus a note on how it was reduced."""
        if a.ndim == 3:  # (C, H, W) -> mean |activation| over channels
            return np.abs(a).mean(0), "mean |activation| over channels"
        if a.ndim == 2:  # (tokens, dim) -> token x feature heatmap
            return a, "tokens × features"
        if a.ndim == 1:
            return a[None], "vector"
        return a.reshape(a.shape[0], -1), "flattened"

    def gallery(self, steps: Optional[Sequence[int]] = None, cols: int = 4,
                title: str = "What each step produces") -> None:
        """Grid of pictures, one per recorded step, titled with the module and its output shape."""
        chosen = [self.steps[i] for i in steps] if steps is not None else self.steps
        if not chosen:
            print("nothing recorded")
            return
        rows = -(-len(chosen) // cols)
        fig = new_figure((3.3 * cols, 3.0 * rows + 0.5))
        for i, s in enumerate(chosen):
            ax = fig.add_subplot(rows, cols, i + 1)
            img, _ = self._summary_image(s.output)
            ax.imshow(img, cmap=MAGNITUDE_CMAP, aspect="auto" if s.output.ndim == 2 else "equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{s.index}. {s.name[-22:]}\n{s.kind} {s.out_shape[1:]}", fontsize=8)
        heading(fig, title)
        display_figure(fig)

    def film(self, steps: Optional[Sequence[int]] = None, channels: int = 6,
             title: str = "Following one image through the network") -> None:
        """GIF: one frame per step - the summary picture, a few channels and a progress bar."""
        chosen = [self.steps[i] for i in steps] if steps is not None else self.steps
        anim = Animator(figsize=(11, 4.4), ms_per_frame=1000)
        for s in chosen:
            with anim.frame() as fig:
                heading(fig, title)
                gs = fig.add_gridspec(2, channels + 2, height_ratios=[5, 1])
                ax = fig.add_subplot(gs[0, :2])
                img, how = self._summary_image(s.output)
                ax.imshow(img, cmap=MAGNITUDE_CMAP, aspect="auto" if s.output.ndim == 2 else "equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(how, fontsize=8)
                if s.output.ndim == 3:
                    for c in range(min(channels, s.output.shape[0])):
                        cax = fig.add_subplot(gs[0, 2 + c])
                        ch = s.output[c]
                        m = float(np.abs(ch).max()) or 1.0
                        cax.imshow(ch, cmap="RdBu_r", vmin=-m, vmax=m)
                        cax.set_xticks([])
                        cax.set_yticks([])
                        cax.set_title(f"ch {c}", fontsize=7)
                bar = fig.add_subplot(gs[1, :])
                n = len(chosen)
                pos = chosen.index(s)
                for j in range(n):
                    face = COLORS["done"] if j < pos else (COLORS["active"] if j == pos
                                                           else COLORS["muted"])
                    bar.add_patch(Rectangle((j, 0), 0.9, 1, facecolor=face))
                bar.set_xlim(0, n)
                bar.set_ylim(0, 1)
                bar.axis("off")
                caption(fig, f"step {s.index}: {s.name} ({s.kind})   "
                             f"{s.in_shape} -> {s.out_shape}", y=0.30)
        anim.show()
