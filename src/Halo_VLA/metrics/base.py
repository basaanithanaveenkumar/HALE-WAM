"""
Base class for all HALE-WAM evaluation metrics.

Design
------
All metrics follow a stateful accumulator pattern:
  1. ``reset()``  — clear accumulators at the start of each eval epoch.
  2. ``update()`` — receive one batch of predictions/targets and accumulate.
  3. ``compute()``— flush accumulators into a ``{name: float}`` dict.

This matches the pattern used across HaleBlocks, HaloDiffusionLLM, and
Halos-lift so metrics can be composed uniformly in trainer eval loops.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RoboticsMetricBase(ABC):
    """Abstract base for HALE-WAM evaluation metrics.

    Every concrete metric must implement the three lifecycle methods below.
    Subclasses should document their ``update`` signature explicitly — the
    kwargs are intentionally open so heterogeneous metrics can be mixed in a
    single eval loop without needing a shared batch schema.
    """

    #: Short identifier used as the prefix for all keys returned by ``compute``.
    name: str = ""

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal accumulators.  Called at the start of each eval epoch."""
        ...

    @abstractmethod
    def update(self, **kwargs) -> None:  # type: ignore[override]
        """Consume one batch of data and update internal state.

        Concrete implementations should replace ``**kwargs`` with explicit
        typed arguments in their signature for IDE discoverability.
        """
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return a flat ``{metric_name: scalar}`` dict from accumulated state.

        Keys should be fully qualified (e.g. ``"action_l1"``, ``"success_rate"``)
        so they can be passed directly to a logger without name collisions.
        """
        ...
