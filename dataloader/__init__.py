"""
Data loading utilities for Halo-VLA.

Stage-aware training:

    from dataloader.stages import TrainingStage, build_stage_dataloader
    loader = build_stage_dataloader(TrainingStage.PRETRAIN, batch_size=256)

Individual dataset loaders:

    from dataloader import EODataset, AiroaMomaDataset
"""

from .airoa_moma_dataset import AiroaMomaDataset, build_airoa_moma_dataloader
from .eo_dataset import EODataset, build_eo_dataloader

__all__ = [
    # Single-dataset loaders (existing API — unchanged)
    "EODataset",
    "build_eo_dataloader",
    "AiroaMomaDataset",
    "build_airoa_moma_dataloader",
    # Stage-aware API (new)
    "TrainingStage",
    "build_stage_dataloader",
]


def __getattr__(name: str):
    if name in ("TrainingStage", "build_stage_dataloader"):
        from dataloader.stages import TrainingStage, build_stage_dataloader  # noqa: PLC0415

        _map = {
            "TrainingStage": TrainingStage,
            "build_stage_dataloader": build_stage_dataloader,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
