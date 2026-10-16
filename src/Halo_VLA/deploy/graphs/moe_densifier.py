"""Swap every ``DeepseekMoE`` of a model for its export form (:class:`DenseMoE`)."""

from __future__ import annotations

from loguru import logger
from torch import nn

from Halo_VLA.deploy.graphs.dense_moe import DenseMoE


class MoEDensifier:
    """In-place module surgery applied before a graph is traced.

    The swap happens on the adapter's private copy of the model (built for export only), keeps
    every weight tensor shared (no copy) and is idempotent.
    """

    def apply(self, model: nn.Module) -> int:
        from models.moe import DeepseekMoE

        replaced = 0
        for parent in list(model.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, DeepseekMoE):
                    setattr(parent, name, DenseMoE(child))
                    replaced += 1
        if replaced:
            logger.info("replaced {} DeepseekMoE layers with DenseMoE for export", replaced)
        return replaced
