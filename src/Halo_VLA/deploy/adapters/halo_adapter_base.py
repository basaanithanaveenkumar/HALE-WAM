"""Shared machinery of every Halo-VLA deployment adapter (Template Method)."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from hale_core.deploy.adapters.export_module import ExportModule
from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import ConfigError
from Halo_VLA.deploy.graphs.moe_densifier import MoEDensifier
from Halo_VLA.deploy.model_factory import HaloModelFactory

if TYPE_CHECKING:
    from config.model_config import HaloVLMConfig
    from dual_system.config import DualSystemConfig


class HaloAdapterBase(ModelAdapter):
    """Build the full training-time network, export one traceable sub-graph of it.

    ``build_model`` returns the complete ``HaloVLM`` / ``DualSystemVLA`` so training checkpoints
    (``model_state_dict``) load with their original keys; :meth:`export_module` then wraps only
    the sub-graph a deployment target runs (:meth:`graph`), so unrelated heads never end up in
    an engine.

    Subclasses implement :meth:`graph` and :meth:`default_signature`.

    Args:
        model: ``HaloVLMConfig`` keyword arguments (the trained architecture).
        dual_system: dual-system preset (``none`` = single-system Halo-VLA).
        dual_system_options: preset overrides.
        max_batch: largest batch the exported graph must accept.
        seed: initialisation seed (without a checkpoint).
        signature / checkpoint / strict: see :class:`ModelAdapter`.
    """

    requires_dual: bool | None = None  # True: dual-system only, False: single-system only

    def __init__(self, model: Mapping[str, Any] | None = None, dual_system: str = "none",
                 dual_system_options: Mapping[str, Any] | None = None, max_batch: int = 4, seed: int = 0,
                 **base: Any) -> None:
        super().__init__(**base)
        self.factory = HaloModelFactory(model, dual_system, dual_system_options, seed)
        self.max_batch = int(max_batch)
        if self.requires_dual is True and not self.factory.is_dual:
            raise ConfigError(f"{type(self).__name__} exports a dual-system part: set adapter.dual_system")
        if self.requires_dual is False and self.factory.is_dual:
            raise ConfigError(f"{type(self).__name__} exports the single-system model: use the DualSystem* "
                              f"adapters for dual_system={self.factory.dual_system!r}")
        self._built: nn.Module | None = None

    # ------------------------------------------------------------------ model
    def build_model(self) -> nn.Module:
        """Built once and cached: the signature of LLM-backed models needs the built config."""
        if self._built is None:
            self._built = self.factory.build()
        return self._built

    def vlm(self, model: nn.Module) -> nn.Module:
        """The ``HaloVLM`` inside ``model`` (itself, or System 2's trunk)."""
        return model.system2.model if self.factory.is_dual else model

    def model_config(self) -> HaloVLMConfig:
        """``HaloVLMConfig`` as the built model sees it (pretrained LLMs rewrite token ids)."""
        if self.factory.model_kwargs.get("llm_backbone"):
            return self.vlm(self.build_model()).config
        return self.factory.config()

    def dual_config(self) -> DualSystemConfig:
        """The resolved dual-system settings (``requires_dual`` adapters only)."""
        from dual_system import resolve_dual_system_config  # lazy: import roots are set up by Halo_VLA.deploy

        return resolve_dual_system_config(self.factory.dual_system, self.factory.dual_system_options)

    # ------------------------------------------------------------------ export
    @abstractmethod
    def graph(self, model: nn.Module) -> nn.Module:
        """The traceable sub-graph of ``model`` to export."""

    def export_module(self, model: nn.Module) -> ExportModule:
        """Densify MoE layers (static shapes), then wrap :meth:`graph` for the exporters."""
        MoEDensifier().apply(model)
        sig = self.signature()
        return ExportModule(self.graph(model).eval(), self.forward_for_export, sig.input_names, sig.outputs).eval()

    def forward_for_export(self, graph: nn.Module, inputs: dict[str, torch.Tensor]) -> Any:
        """Call ``graph`` with the named inputs (default: keyword arguments)."""
        return graph(**inputs)

    def calibration_batches(self, num_batches: int) -> Iterator[dict[str, torch.Tensor]]:
        """Template-shaped samples (placeholders in place) at the ``opt`` shapes, one seed per batch."""
        sig = self.signature()
        for i in range(num_batches):
            yield sig.sample("opt", torch.Generator().manual_seed(1000 + i))

    def batch_dim(self) -> list[int]:
        """The batch axis as ``[min, opt, max]``."""
        return [1, 1, self.max_batch]

    @staticmethod
    def build_signature(cfg: Mapping[str, Any]) -> IOSignature:
        """Build an :class:`IOSignature` from a plain mapping (inputs, outputs, dims)."""
        return IOSignature.from_config(cfg)
