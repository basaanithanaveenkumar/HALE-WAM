"""Phase 1 — Pre-training dataloaders for Halo-VLA.

Pre-training uses web-scale video/image data and large open-world robot
teleoperation corpora to learn generalised visual-motor representations.

Datasets registered here:
  - ``eo-data-interleave``  : EO-Data1.5M interleaved subsets (web robot QA)
  - ``open-x-pretrain``     : Open X-Embodiment large-scale mix
  - ``bridge-v2``           : Bridge Data V2 (60K diverse manipulation demos)
  - ``fractal-rt1``         : Google RT-1 training corpus (130K episodes)

Usage::

    from dataloader.stages.pretrain import PretrainDatasetRegistry

    registry = PretrainDatasetRegistry(batch_size=32)
    loader = registry.build()
    for batch in loader:
        images, actions, states, texts = (
            batch["images"], batch["actions"], batch["states"], batch["text"],
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from loguru import logger

# ---------------------------------------------------------------------------
# Pre-training dataset specs
# ---------------------------------------------------------------------------

PRETRAIN_DATASETS: dict[str, dict[str, Any]] = {
    "eo-data-interleave": {
        "hf_path": "IPEC-COMMUNITY/EO-Data1.5M",
        "subsets": [
            "interleave-free_chat",
            "interleave-temporal",
            "interleave-trajectory",
            "interleave-video_caption",
        ],
        "description": (
            "EO-Data1.5M interleaved subsets — "
            "web-scale robot observation QA across 4 interleaved categories."
        ),
        "paper_reference": "EO-Data1.5M (IPEC-COMMUNITY, 2024)",
        "n_episodes_approx": 1_500_000,
    },
    "open-x-pretrain": {
        "hf_path": "jxu124/OpenX-Embodiment",
        "subsets": None,
        "description": (
            "Open X-Embodiment — 22 robot types, ~2M demonstrations. "
            "Broadest available robot pretraining corpus."
        ),
        "paper_reference": "Open X-Embodiment Collaboration (2023)",
        "n_episodes_approx": 2_000_000,
    },
    "bridge-v2": {
        "hf_path": "lerobot/bridge_v2",
        "subsets": None,
        "description": (
            "Bridge Data V2 — ~60K diverse household robot manipulation demos "
            "across 24 environments."
        ),
        "paper_reference": "Walke et al. (2023) Bridge Data V2",
        "n_episodes_approx": 60_000,
    },
    "fractal-rt1": {
        "hf_path": "google-deepmind/fractal20220817-data",
        "subsets": None,
        "description": (
            "Google RT-1 training corpus — 130K episodes across 700+ task "
            "variants on a mobile manipulation robot."
        ),
        "paper_reference": "Brohan et al. (2022) RT-1",
        "n_episodes_approx": 130_000,
    },
}

PRETRAIN_DATASET_NAMES: tuple[str, ...] = tuple(PRETRAIN_DATASETS)


@dataclass
class PretrainDatasetConfig:
    """Configuration for the pre-training dataloader."""

    dataset_names: list[str] = field(default_factory=lambda: list(PRETRAIN_DATASET_NAMES))
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32
    max_samples_per_dataset: int | None = None  # None = no cap
    streaming: bool = True
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True


class PretrainDatasetRegistry:
    """Factory that assembles a pre-training DataLoader from the registered datasets.

    The registry streams each dataset lazily from HuggingFace Hub (no full
    downloads required). For local datasets pass ``hf_path`` as a local dir.
    """

    def __init__(self, **config_overrides) -> None:
        self.cfg = PretrainDatasetConfig(**config_overrides)

    def build(self) -> DataLoader:
        """Build and return a DataLoader covering all registered pretrain datasets."""
        from dataloader.eo_dataset import EODataset, EODatasetConfig, build_eo_dataloader

        datasets = []
        for name in self.cfg.dataset_names:
            if name not in PRETRAIN_DATASETS:
                logger.warning("pretrain dataset {!r} not registered — skipping", name)
                continue
            spec = PRETRAIN_DATASETS[name]
            logger.info(
                "adding pretrain dataset name={} hf_path={} episodes≈{}",
                name,
                spec["hf_path"],
                spec["n_episodes_approx"],
            )

            if name == "eo-data-interleave":
                # EO-Data has its own loader; build one per requested subset
                for subset in (spec["subsets"] or ["interleave-temporal"]):
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
                # Generic HF dataset wrapper
                datasets.append(
                    _HFRobotDataset(
                        name=name,
                        spec=spec,
                        cfg=self.cfg,
                    )
                )

        if not datasets:
            raise RuntimeError("No pretrain datasets available — check dataset_names config.")

        combined = ConcatDataset(datasets)
        from dataloader.eo_dataset import eo_collate_fn

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

    def __init__(
        self,
        *,
        name: str,
        spec: dict[str, Any],
        cfg: PretrainDatasetConfig,
    ) -> None:
        self.name = name
        self.spec = spec
        self.cfg = cfg
        self._data: list[dict] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        logger.info("loading pretrain dataset {} from {}", self.name, self.spec["hf_path"])
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
    "PRETRAIN_DATASET_NAMES",
    "PRETRAIN_DATASETS",
    "PretrainDatasetConfig",
    "PretrainDatasetRegistry",
]
