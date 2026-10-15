"""``eval``: held-out validation (losses + action metrics, ``training/evaluator.py``)."""

from __future__ import annotations

from config.sections.base import VLASection


class VLAEvalConfig(VLASection):
    ARGS = {"val_fraction": "val_fraction", "val_seed": "val_seed", "every_n_epochs": "eval_every",
            "max_batches": "eval_max_batches", "flow_steps": "eval_flow_steps"}

    val_fraction: float = 0.0              # share of training samples held out (0 = off)
    val_seed: int = 0
    every_n_epochs: int = 1
    max_batches: int | None = None
    flow_steps: int | None = None          # Euler steps when sampling actions
