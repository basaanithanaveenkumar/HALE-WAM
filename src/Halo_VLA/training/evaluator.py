"""Validation for Halo-VLA: held-out losses plus action metrics (counterpart of Hale-VLM's evaluator)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from torch.amp import autocast

from metrics import ActionChunkAccuracyMetric, ActionL1Metric, ActionMSEMetric, TrajectorySmoothnessMetric
from training.training_step import HaloVLATrainingStep


class VLAEvaluator:
    """Runs the model on a validation loader and reports losses and action quality.

    For every batch it computes the same three losses as training (language, action flow,
    visual) and, from the decoder states at ``<halo_action>`` tokens, samples action chunks with
    the flow decoder (``HaloVLM.sample_actions``, Euler ODE) and scores them against the
    ground-truth actions:

    * ``val/action_l1`` (+ first three timesteps), ``val/action_mse`` / ``val/action_rmse``;
    * ``val/chunk_acc_*``: share of chunks whose worst-case L1 error is under each threshold;
    * ``val/pred_smoothness`` vs ``val/target_smoothness``: jerk proxy of the trajectories.

    Per-dimension L1 values are kept out of the summary (``action_l1_dim_*``) to keep logs
    readable; :meth:`run` with ``per_dim=True`` includes them.

    Args:
        action_dim: action vector size (``None`` = read from the first batch's ``actions``).
        max_batches: stop after this many batches (``None`` = whole loader).
        flow_steps: Euler steps for action sampling (``None`` = the model's default).
    """

    def __init__(self, action_dim: int | None = None, max_batches: int | None = None, flow_steps: int | None = None) -> None:
        self.action_dim = action_dim
        self.max_batches = max_batches
        self.flow_steps = flow_steps

    @torch.no_grad()
    def run(
        self,
        step: HaloVLATrainingStep,
        loader: Iterable[dict],
        device: torch.device,
        *,
        amp: bool = False,
        per_dim: bool = False,
    ) -> dict[str, float]:
        """Metrics over ``loader`` (call on rank 0 with the *unwrapped* training step)."""
        model = step.model
        was_training = model.training
        model.eval()
        metrics: list = []  # built on the first batch, sized from its actions
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        try:
            for index, batch in enumerate(loader):
                if self.max_batches is not None and index >= self.max_batches:
                    break
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast(device.type, enabled=amp):
                    out = step(batch)
                for name in ("total", "lang", "act", "vis"):
                    sums[f"loss_{name}"] += float(getattr(out, name))
                batches += 1
                if not metrics:
                    metrics.extend(self._make_metrics(int(batch["actions"].shape[-1])))
                self._score_actions(model, out.action_hiddens, batch, metrics)
        finally:
            model.train(was_training)
        if batches == 0:
            raise ValueError("the validation loader produced no batches")
        result = {f"val/{k}": v / batches for k, v in sums.items()}
        for metric in metrics:
            for key, value in metric.compute().items():
                if per_dim or "_dim_" not in key:
                    result[f"val/{key}"] = float(value)
        return result

    def _make_metrics(self, data_dim: int) -> list:
        dim = self.action_dim or data_dim
        if dim != data_dim:
            raise ValueError(f"evaluator action_dim={dim} but the data has {data_dim}-dim actions")
        return [ActionL1Metric(dim), ActionMSEMetric(dim), ActionChunkAccuracyMetric(), TrajectorySmoothnessMetric()]

    def _score_actions(self, model, action_hiddens, batch: dict, metrics: list) -> None:
        """Sample action chunks and update the metrics on the valid (unpadded) timesteps."""
        if hasattr(model, "predict_actions"):  # dual-system VLA: System 2 -> latent -> System 1
            preds = model.predict_actions(batch).unsqueeze(1)
        elif action_hiddens is None:
            return
        else:
            preds = model.sample_actions(action_hiddens.float(), num_steps=self.flow_steps)  # [B, n, chunk, D]
        b = preds.shape[0]
        preds = preds.reshape(b, -1, preds.shape[-1])  # [B, n * chunk, D]
        target = batch["actions"].float()
        steps = min(preds.shape[1], target.shape[1])
        pred, target = preds[:, :steps], target[:, :steps]
        mask = batch["action_mask"][:, :steps].bool()
        l1, mse, chunk, smooth = metrics
        l1.update(pred=pred, target=target, mask=mask)
        mse.update(pred=pred, target=target, mask=mask)
        # Padded steps carry no target: zero their error so chunk accuracy only judges real steps.
        keep = mask.unsqueeze(-1)
        chunk.update(pred=pred * keep, target=target * keep)
        smooth.update(pred=pred, target=target)
