"""``train``: optimisation, checkpoints and the Ray Train launch."""

from __future__ import annotations

from pydantic import Field

from config.sections.base import VLASection


class CheckpointConfig(VLASection):
    ARGS = {"dir": "ckpt_dir", "every_n_steps": "save_every", "keep_last": "keep_ckpts"}

    dir: str = "checkpoints"
    every_n_steps: int = 100
    keep_last: int = 3                     # 0 = keep all


class RaySection(VLASection):
    """Launch on Ray Train workers (DDP inside every worker; docs/distributed.md)."""

    ARGS = {"enabled": "ray", "address": "ray_address", "num_workers": "ray_num_workers", "use_gpu": "ray_use_gpu",
            "cpus_per_worker": "ray_cpus_per_worker", "storage_path": "ray_storage_path",
            "max_failures": "ray_max_failures"}

    enabled: bool = False
    address: str | None = None             # 'auto' / ray://host:10001; None starts a local Ray
    num_workers: int = 1
    use_gpu: bool = False
    cpus_per_worker: float | None = None
    storage_path: str = "ray_results"
    max_failures: int = 0


class VLATrainConfig(VLASection):
    ARGS = {"epochs": "epochs", "batch_size": "batch_size", "lr": "lr", "weight_decay": "weight_decay",
            "grad_clip": "grad_clip", "grad_accum_steps": "grad_accum_steps",
            "action_loss_weight": "action_loss_weight", "resume": "resume"}

    epochs: int = 5
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    action_loss_weight: float = 1.0
    resume: str | None = None              # checkpoint .pt to resume from
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    ray: RaySection = Field(default_factory=RaySection)
