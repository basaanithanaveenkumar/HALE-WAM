"""stepviz - pictures and animations for the from-scratch notebooks.

Every from-scratch notebook explains an architecture or algorithm one step at a time; this kit gives
each step a picture:

* :mod:`stepviz.tensors` - images, feature-map mosaics, kernels, heatmaps, histograms, curves
* :mod:`stepviz.ops` - hand-computed animations of conv, pooling, patchify and attention
* :mod:`stepviz.recorder` - ``StepRecorder``: hooks that picture every layer of any ``nn.Module``
* :mod:`stepviz.diagrams` - the tensor-journey figure and labelled block diagrams
* :class:`Animator` - draw frames with ``with anim.frame() as fig:`` and embed them as a GIF

It depends only on numpy, matplotlib, Pillow and (for the recorder) PyTorch, so the notebooks stay
"pure Python / PyTorch". The canonical copy lives in ``HaleBlocks/notebooks/_from_scratch_kit``; the
build script syncs it into every other Hale repository.
"""

from . import diagrams, ops, tensors
from ._core import COLORS, Animator, caption, display_figure, heading, new_figure
from .recorder import Step, StepRecorder

__all__ = [
    "COLORS", "Animator", "caption", "display_figure", "heading", "new_figure",
    "Step", "StepRecorder", "diagrams", "ops", "tensors",
]
