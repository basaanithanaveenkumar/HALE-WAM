"""RaySection: launch on Ray Train workers (DDP inside every worker; docs/distributed.md)."""

from __future__ import annotations

from config.sections.base import VLASection


class RaySection(VLASection):
    """Launch on Ray Train workers (DDP inside every worker; docs/distributed.md)."""

    ARGS = {
        "enabled": "ray",
        "address": "ray_address",
        "num_workers": "ray_num_workers",
        "use_gpu": "ray_use_gpu",
        "cpus_per_worker": "ray_cpus_per_worker",
        "storage_path": "ray_storage_path",
        "max_failures": "ray_max_failures",
    }

    enabled: bool = False
    address: str | None = None  # 'auto' / ray://host:10001; None starts a local Ray
    num_workers: int = 1
    use_gpu: bool = False
    cpus_per_worker: float | None = None
    storage_path: str = "ray_results"
    max_failures: int = 0
