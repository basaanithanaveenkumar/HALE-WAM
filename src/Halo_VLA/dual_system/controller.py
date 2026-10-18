"""Asynchronous inference: System 2 updates the latent slowly, System 1 acts every step."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import torch

from dual_system.model import DualSystemVLA


class DualSystemController:
    """Runs a :class:`DualSystemVLA` at two frequencies.

    Every call to :meth:`act` is one control step: System 1 always runs on the newest
    observation; System 2 refreshes the latent only every ``async_steps`` calls. Between
    refreshes System 1 reuses the last latent - the "temporal delay" of dual-process control.

    * ``threaded=False`` (default, deterministic): System 2 runs inline on refresh steps.
    * ``threaded=True``: System 2 runs in a background thread; System 1 keeps acting on the
      previous latent until the new one is ready, so the control loop never waits for the
      slow model (after the very first latent).

    Args:
        model: the dual-system VLA (eval mode is set here).
        async_steps: System 1 steps per System 2 update (default: the model's config).
        threaded: overlap System 2 with System 1 in a background thread.
    """

    def __init__(self, model: DualSystemVLA, async_steps: int | None = None, threaded: bool = False) -> None:
        self.model = model.eval()
        self.async_steps = async_steps or model.config.async_steps
        self.threaded = threaded
        self._pool = ThreadPoolExecutor(max_workers=1) if threaded else None
        self.reset()

    def reset(self) -> None:
        """Start a new episode (drops the cached latent)."""
        self.step = 0
        self.latent: tuple[torch.Tensor, torch.Tensor] | None = None
        self.pending: Future | None = None
        self.system2_calls = 0

    def _encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            latent, mask, _ = self.model.encode_latent(batch)
        return latent, mask

    def _refresh(self, batch: dict) -> None:
        self.system2_calls += 1
        if not self.threaded or self.latent is None:
            self.latent = self._encode(batch)  # the first latent is always computed inline
            return
        if self.pending is None or self.pending.done():
            self.pending = self._pool.submit(self._encode, batch)

    def act(self, batch: dict) -> torch.Tensor:
        """One control step: an action chunk ``(B, chunk, action_dim)`` for the observation."""
        if self.pending is not None and self.pending.done():
            self.latent, self.pending = self.pending.result(), None
        if self.latent is None or self.step % self.async_steps == 0:
            self._refresh(batch)
        self.step += 1
        return self.model.predict_actions(batch, latent=self.latent)

    def close(self) -> None:
        """Wait for any pending System 2 update and stop the worker thread."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
