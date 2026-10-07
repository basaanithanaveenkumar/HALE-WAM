"""Phase 2 — Mid-training dataloaders for Halo-VLA.

Mid-training uses domain-specific robot manipulation data to adapt pretrained
representations to the target workspace and object distribution.

Datasets registered here:
  - ``droid``       : DROID full Franka corpus (76K diverse demos)
  - ``airoa-moma``  : AIRoA-MoMA HSR teleoperation (RGB video + metadata)
  - ``eo-data-qa``  : EO-Data1.5M QA subsets (12 task-structured categories)
  - ``rh20t``       : RH20T contact-rich manipulation (110K demos)

Usage::

    from dataloader.stages.mid_train import MidTrainDatasetRegistry

    registry = MidTrainDatasetRegistry(batch_size=64)
    loader = registry.build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from loguru import logger


# ---------------------------------------------------------------------------
# Mid-training dataset specs
# ---------------------------------------------------------------------------

MID_TRAIN_DATASETS: dict[str, dict[str, Any]] = {
    "droid": {
        "hf_path": "lerobot/droid_100",
        "description": (
            "DROID — 76K diverse Franka arm demonstrations across labs. "
            "Collected with three cameras (2 exterior, 1 wrist)."
        ),
        "paper_reference": "Khazatsky et al. (2024) DROID",
        "n_episodes_approx": 76_000,
        "dataset_cls": "DroidDataset",
    },
    "airoa-moma": {
        "hf_path": "airoa-org/airoa-moma",
        "description": (
            "AIRoA-MoMA — HSR teleoperation with RGB video and metadata. "
            "Local clone required after `git lfs pull`."
        ),
        "paper_reference": "AIRoA-MoMA (2024)",
        "n_episodes_approx": None,  # depends on local clone
        "dataset_cls": "AiroaMomaDataset",
    },
    "eo-data-qa": {
        "hf_path": "IPEC-COMMUNITY/EO-Data1.5M",
        "subsets": [
            "qa-affordance_qa",
            "qa-episode_caption",
            "qa-failure_detection",
            "qa-multiview_qa",
            "qa-subtask_qa",
            "qa-task_planning",
            "qa-trajectory_qa",
        ],
        "description": "EO-Data1.5M QA subsets — 7 structured task-comprehension categories.",
        "paper_reference": "EO-Data1.5M (IPEC-COMMUNITY, 2024)",
        "n_episodes_approx": 700_000,
        "dataset_cls": "EODataset",
    },
    "rh20t": {
        "hf_path": "lerobot/rh20t",
        "description": (
            "RH20T — 110K contact-rich multi-modal robot manipulation demos "
            "with force/torque sensing."
        ),
        "paper_reference": "Fang et al. (2023) RH20T",
        "n_episodes_approx": 110_000,
        "dataset_cls": None,  # generic HF wrapper
    },
}

MID_TRAIN_DATASET_NAMES: tuple[str, ...] = tuple(MID_TRAIN_DATASETS)


@dataclass
class MidTrainDatasetConfig:
    """Configuration for the mid-training dataloader."""

    dataset_names: list[str] = field(default_factory=lambda: list(MID_TRAIN_DATASET_NAMES))
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32
    max_samples_per_dataset: int | None = 50_000
    streaming: bool = True
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    # DROID-specific
    droid_data_root: str | None = None
    droid_num_sample_frames: int = 1
    droid_num_predict_frames: int = 1
    # AIRoA-MoMA-specific
    airoa_data_root: str | None = None


class MidTrainDatasetRegistry:
    """Factory that assembles a mid-training DataLoader from registered datasets."""

    def __init__(self, **config_overrides) -> None:
        self.cfg = MidTrainDatasetConfig(**config_overrides)

    def build(self) -> DataLoader:
        """Build and return a DataLoader covering all registered mid-train datasets."""
        from dataloader.eo_dataset import EODataset, EODatasetConfig, eo_collate_fn

        datasets = []
        for name in self.cfg.dataset_names:
            if name not in MID_TRAIN_DATASETS:
                logger.warning("mid-train dataset {!r} not registered — skipping", name)
                continue
            spec = MID_TRAIN_DATASETS[name]
            logger.info("adding mid-train dataset name={}", name)

            if name == "droid" and self.cfg.droid_data_root:
                from dataloader.droid_dataset import DroidDataset

                datasets.append(
                    DroidDataset(
                        data_root=self.cfg.droid_data_root,
                        img_size=self.cfg.img_size,
                        num_sample_frames=self.cfg.droid_num_sample_frames,
                        num_predict_frames=self.cfg.droid_num_predict_frames,
                    )
                )
            elif name == "airoa-moma" and self.cfg.airoa_data_root:
                from dataloader.airoa_moma_dataset import AiroaMomaDataset

                datasets.append(
                    AiroaMomaDataset(
                        data_root=self.cfg.airoa_data_root,
                        img_size=self.cfg.img_size,
                    )
                )
            elif name == "eo-data-qa":
                for subset in spec.get("subsets", ["qa-task_planning"]):
                    cfg = EODatasetConfig(
                        dataset_name=spec["hf_path"],
                        subset=subset,
                        split=self.cfg.split,
                        img_size=self.cfg.img_size,
                        max_seq_len=self.cfg.max_seq_len,
                        max_action_len=self.cfg.max_action_len,
                        action_dim=self.cfg.action_dim,
                    )
                    datasets.append(EODataset(cfg))
            else:
                datasets.append(
                    _HFRobotDataset(
                        name=name,
                        spec=spec,
                        cfg=self.cfg,
                    )
                )

        if not datasets:
            raise RuntimeError(
                "No mid-train datasets available. "
                "Provide droid_data_root / airoa_data_root for local datasets."
            )

        combined = ConcatDataset(datasets)
        return DataLoader(
            combined,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=eo_collate_fn,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )


class _HFRobotDataset(torch.utils.data.Dataset):
    """Thin wrapper that lazy-loads a HuggingFace robot dataset."""

    def __init__(self, *, name: str, spec: dict[str, Any], cfg: MidTrainDatasetConfig) -> None:
        self.name = name
        self.spec = spec
        self.cfg = cfg
        self._data: list[dict] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        logger.info("loading mid-train dataset {} from {}", self.name, self.spec["hf_path"])
        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.spec["hf_path"],
                split=self.cfg.split,
                streaming=self.cfg.streaming,
                trust_remote_code=True,
            )
            if self.cfg.streaming:
                max_n = self.cfg.max_samples_per_dataset or 10_000
                self._data = list(ds.take(max_n))
            else:
                cap = self.cfg.max_samples_per_dataset
                self._data = list(ds) if cap is None else list(ds.select(range(min(cap, len(ds)))))
        except Exception as exc:
            logger.error("failed to load {} — {}", self.name, exc)
            self._data = []

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._data)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        self._ensure_loaded()
        return self._data[idx]  # type: ignore[index]


__all__ = [
    "MID_TRAIN_DATASET_NAMES",
    "MID_TRAIN_DATASETS",
    "MidTrainDatasetConfig",
    "MidTrainDatasetRegistry",
]
