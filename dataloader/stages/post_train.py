"""Phase 3 — Post-training dataloaders for Halo-VLA.

Post-training uses small, task-specific demonstration and instruction-tuning
sets to maximise metric performance on target benchmarks.

VLM datasets (instruction-following fine-tuning):
  - ``llava-instruct-665k``   : LLaVA Instruct 665K multi-turn SFT
  - ``textvqa``                : TextVQA 28K reading-comprehension VQA
  - ``scienceqa``              : ScienceQA 21K multimodal science QA
  - ``chartqa``                : ChartQA 20K chart reasoning
  - ``infographics-vqa``       : InfographicsVQA 30K document VQA
  - ``seed-bench``             : SEED-Bench 19K multi-choice VQA
  - ``llava-plus``             : LLaVA-Plus tool-augmented SFT

VLA datasets (robot task fine-tuning):
  - ``droid-task``             : DROID task-specific subset (curated per task)
  - ``eo-data-task``           : EO-Data1.5M task-targeted subsets
  - ``airoa-task``             : AIRoA-MoMA task-filtered episodes
  - ``libero-goal``            : LIBERO-Goal 10-task goal-conditioned benchmark
  - ``libero-spatial``         : LIBERO-Spatial 10-task spatial reasoning
  - ``libero-object``          : LIBERO-Object 10-task object-manipulation
  - ``libero-100``             : LIBERO-100 full 100-task benchmark
  - ``aloha-bimanual``         : ALOHA real bimanual demos (200 eps)
  - ``metaworld-mt50``         : Meta-World MT50 (50 tasks, 2,500 eps)

Usage::

    from dataloader.stages.post_train import PostTrainDatasetRegistry

    registry = PostTrainDatasetRegistry(
        task_filter="pick_and_place",
        batch_size=32,
    )
    loader = registry.build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader

from loguru import logger


# ---------------------------------------------------------------------------
# Post-training dataset specs
# ---------------------------------------------------------------------------

POST_TRAIN_DATASETS: dict[str, dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # VLM instruction-following fine-tuning datasets
    # -----------------------------------------------------------------------
    "llava-instruct-665k": {
        "hf_path": "liuhaotian/LLaVA-Instruct-150K",
        "description": "LLaVA Instruct 665K — multi-turn conversation SFT with GPT-4.",
        "paper_reference": "Liu et al. (2023) LLaVA",
        "n_episodes_approx": 665_000,
        "data_type": "vlm",
    },
    "textvqa": {
        "hf_path": "lmms-lab/textvqa",
        "description": "TextVQA — 28K reading-comprehension VQA on scene text.",
        "paper_reference": "Singh et al. (2019) TextVQA",
        "n_episodes_approx": 28_000,
        "data_type": "vlm",
    },
    "scienceqa": {
        "hf_path": "derek-thomas/ScienceQA",
        "description": "ScienceQA — 21K multi-modal science questions with explanations.",
        "paper_reference": "Lu et al. (2022) ScienceQA",
        "n_episodes_approx": 21_208,
        "data_type": "vlm",
    },
    "chartqa": {
        "hf_path": "HuggingFaceM4/ChartQA",
        "description": "ChartQA — 20K chart-based reasoning questions.",
        "paper_reference": "Masry et al. (2022) ChartQA",
        "n_episodes_approx": 20_000,
        "data_type": "vlm",
    },
    "infographics-vqa": {
        "hf_path": "jordyvl/DocVQA_InfographicsVQA_questions_answers",
        "description": "InfographicsVQA — 30K document and infographic VQA pairs.",
        "paper_reference": "Mathew et al. (2021) InfographicVQA",
        "n_episodes_approx": 30_035,
        "data_type": "vlm",
    },
    "seed-bench": {
        "hf_path": "AILab-CVC/SEED-Bench",
        "description": "SEED-Bench — 19K multiple-choice VQA spanning 12 dimensions.",
        "paper_reference": "Li et al. (2023) SEED-Bench",
        "n_episodes_approx": 19_242,
        "data_type": "vlm",
    },
    "llava-plus": {
        "hf_path": "liuhaotian/LLaVA-Plus-Data",
        "description": "LLaVA-Plus — tool-augmented multi-step instruction-following SFT.",
        "paper_reference": "Liu et al. (2023) LLaVA-Plus",
        "n_episodes_approx": 100_000,
        "data_type": "vlm",
    },
    # -----------------------------------------------------------------------
    # VLA task-specific fine-tuning datasets
    # -----------------------------------------------------------------------
    "droid-task": {
        "hf_path": "lerobot/droid_100",
        "description": (
            "DROID task-specific subset — filter by task keyword for "
            "targeted fine-tuning on a specific manipulation skill."
        ),
        "paper_reference": "Khazatsky et al. (2024) DROID",
        "n_episodes_approx": 500,  # per task after filtering
        "data_type": "vla",
    },
    "eo-data-task": {
        "hf_path": "IPEC-COMMUNITY/EO-Data1.5M",
        "subsets": [
            "qa-process_verification",
            "qa-relation_reasoning",
            "qa-object_referring_qa",
            "qa-physical_common_sense",
            "qa-points_qa",
        ],
        "description": "EO-Data1.5M task-targeted subsets for physical reasoning fine-tuning.",
        "paper_reference": "EO-Data1.5M (IPEC-COMMUNITY, 2024)",
        "n_episodes_approx": 50_000,
        "data_type": "vla",
    },
    "airoa-task": {
        "hf_path": "airoa-org/airoa-moma",
        "description": (
            "AIRoA-MoMA task-filtered episodes. "
            "Set airoa_data_root to local LFS clone path."
        ),
        "paper_reference": "AIRoA-MoMA (2024)",
        "n_episodes_approx": None,
        "data_type": "vla",
    },
    "libero-goal": {
        "hf_path": "lerobot/libero_goal",
        "description": "LIBERO-Goal — 10 goal-conditioned tasks on a Franka arm (sim).",
        "paper_reference": "Liu et al. (2023) LIBERO",
        "n_episodes_approx": 500,
        "data_type": "vla",
    },
    "libero-spatial": {
        "hf_path": "lerobot/libero_spatial",
        "description": "LIBERO-Spatial — 10 spatial-reasoning tasks on a Franka arm (sim).",
        "paper_reference": "Liu et al. (2023) LIBERO",
        "n_episodes_approx": 500,
        "data_type": "vla",
    },
    "libero-object": {
        "hf_path": "lerobot/libero_object",
        "description": "LIBERO-Object — 10 object-manipulation tasks on a Franka arm (sim).",
        "paper_reference": "Liu et al. (2023) LIBERO",
        "n_episodes_approx": 500,
        "data_type": "vla",
    },
    "libero-100": {
        "hf_path": "lerobot/libero_10",
        "description": "LIBERO-100 — full 100-task benchmark, 5 demo suites on Franka.",
        "paper_reference": "Liu et al. (2023) LIBERO",
        "n_episodes_approx": 1_693,
        "data_type": "vla",
    },
    "aloha-bimanual": {
        "hf_path": "lerobot/aloha_mobile_cabinet",
        "description": "ALOHA real bimanual demos — 200 cabinet-manipulation episodes.",
        "paper_reference": "Zhao et al. (2023) ACT",
        "n_episodes_approx": 200,
        "data_type": "vla",
    },
    "metaworld-mt50": {
        "hf_path": "lerobot/metaworld_mt50",
        "description": "Meta-World MT50 — 50 tasks, 2,500 episodes on a Sawyer arm.",
        "paper_reference": "Yu et al. (2020) Meta-World",
        "n_episodes_approx": 2_500,
        "data_type": "vla",
    },
}

POST_TRAIN_DATASET_NAMES: tuple[str, ...] = tuple(POST_TRAIN_DATASETS)


@dataclass
class PostTrainDatasetConfig:
    """Configuration for the post-training dataloader."""

    dataset_names: list[str] = field(default_factory=lambda: list(POST_TRAIN_DATASET_NAMES))
    split: str = "train"
    img_size: int = 224
    max_seq_len: int = 77
    max_action_len: int = 64
    action_dim: int = 32
    # For task-filtered datasets supply a keyword (e.g. "pick_and_place").
    task_filter: str | None = None
    max_samples_per_dataset: int | None = None  # small datasets — keep all
    streaming: bool = False  # materialise in memory for small task sets
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    # Local paths for gated / large datasets
    droid_data_root: str | None = None
    airoa_data_root: str | None = None


class PostTrainDatasetRegistry:
    """Factory that assembles a post-training DataLoader from registered datasets."""

    def __init__(self, **config_overrides) -> None:
        self.cfg = PostTrainDatasetConfig(**config_overrides)

    def build(self) -> DataLoader:
        """Build and return a DataLoader covering all registered post-train datasets."""
        from dataloader.eo_dataset import EODataset, EODatasetConfig, eo_collate_fn

        datasets = []
        for name in self.cfg.dataset_names:
            if name not in POST_TRAIN_DATASETS:
                logger.warning("post-train dataset {!r} not registered — skipping", name)
                continue
            spec = POST_TRAIN_DATASETS[name]
            logger.info("adding post-train dataset name={}", name)

            if name == "droid-task" and self.cfg.droid_data_root:
                from dataloader.droid_dataset import DroidDataset

                datasets.append(
                    DroidDataset(
                        data_root=self.cfg.droid_data_root,
                        img_size=self.cfg.img_size,
                        task_filter=self.cfg.task_filter,
                    )
                )
            elif name == "airoa-task" and self.cfg.airoa_data_root:
                from dataloader.airoa_moma_dataset import AiroaMomaDataset

                datasets.append(
                    AiroaMomaDataset(
                        data_root=self.cfg.airoa_data_root,
                        img_size=self.cfg.img_size,
                        task_filter=self.cfg.task_filter,
                    )
                )
            elif name == "eo-data-task":
                for subset in spec.get("subsets", ["qa-process_verification"]):
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
                "No post-train datasets available. "
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
            drop_last=False,
        )


class _HFRobotDataset(torch.utils.data.Dataset):
    """Thin wrapper that lazy-loads a HuggingFace robot dataset."""

    def __init__(
        self, *, name: str, spec: dict[str, Any], cfg: PostTrainDatasetConfig
    ) -> None:
        self.name = name
        self.spec = spec
        self.cfg = cfg
        self._data: list[dict] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        logger.info("loading post-train dataset {} from {}", self.name, self.spec["hf_path"])
        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.spec["hf_path"],
                split=self.cfg.split,
                streaming=self.cfg.streaming,
                trust_remote_code=True,
            )
            cap = self.cfg.max_samples_per_dataset
            if self.cfg.streaming:
                self._data = list(ds.take(cap or 5_000))
            else:
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
    "POST_TRAIN_DATASET_NAMES",
    "POST_TRAIN_DATASETS",
    "PostTrainDatasetConfig",
    "PostTrainDatasetRegistry",
]
