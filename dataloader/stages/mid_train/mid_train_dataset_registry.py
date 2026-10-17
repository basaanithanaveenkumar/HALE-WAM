"""MidTrainDatasetRegistry: midTrainDatasetRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from exceptions import DatasetError
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader

if TYPE_CHECKING:  # annotation only; the runtime import is lazy (import cycle)
    from .mid_train_dataset_config import MidTrainDatasetConfig

MID_TRAIN_DATASETS: dict[str, dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # VLM connector-pretraining datasets
    # -----------------------------------------------------------------------
    "llava-pretrain-558k": {
        "hf_path": "liuhaotian/LLaVA-Pretrain",
        "description": "LLaVA Pretrain 558K — curated BLIP captions for connector pretraining.",
        "paper_reference": "Liu et al. (2023) LLaVA",
        "n_episodes_approx": 558_000,
        "data_type": "vlm",
    },
    "sharegpt4v-pt": {
        "hf_path": "Lin-Chen/ShareGPT4V",
        "subsets": ["ShareGPT4V-PT"],
        "description": "ShareGPT4V-PT — 1.2M GPT-4V-generated high-quality captions.",
        "paper_reference": "Chen et al. (2023) ShareGPT4V",
        "n_episodes_approx": 1_200_000,
        "data_type": "vlm",
    },
    "blip-laion-cc-sbu-558k": {
        "hf_path": "liuhaotian/LLaVA-Pretrain",
        "description": "BLIP LAION+CC+SBU 558K filtered pretrain captions.",
        "paper_reference": "Li et al. (2022) BLIP",
        "n_episodes_approx": 558_000,
        "data_type": "vlm",
    },
    "recap-datacomp-1b": {
        "hf_path": "UCSC-VLAA/Recap-DataComp-1B",
        "description": "Recap DataComp-1B — 1.28B LLaMA-recaptioned image-text pairs.",
        "paper_reference": "Li et al. (2024) Recap-DataComp-1B",
        "n_episodes_approx": 1_280_000_000,
        "data_type": "vlm",
    },
    "allava-vflan": {
        "hf_path": "FreedomIntelligence/ALLaVA-4V",
        "subsets": ["allava_vflan"],
        "description": "AllaVA-vFLAN — 1.3M diverse VQA, reasoning and captioning.",
        "paper_reference": "Chen et al. (2024) AllaVA",
        "n_episodes_approx": 1_300_000,
        "data_type": "vlm",
    },
    # -----------------------------------------------------------------------
    # VLA domain-adaptation datasets
    # -----------------------------------------------------------------------
    "droid": {
        "hf_path": "lerobot/droid_100",
        "description": (
            "DROID — 76K diverse Franka arm demonstrations across labs. "
            "Collected with three cameras (2 exterior, 1 wrist)."
        ),
        "paper_reference": "Khazatsky et al. (2024) DROID",
        "n_episodes_approx": 76_000,
        "dataset_cls": "DroidDataset",
        "data_type": "vla",
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
        "data_type": "vla",
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
        "data_type": "vla",
    },
    "rh20t": {
        "hf_path": "lerobot/rh20t",
        "description": (
            "RH20T — 110K contact-rich multi-modal robot manipulation demos "
            "with force/torque sensing."
        ),
        "paper_reference": "Fang et al. (2023) RH20T",
        "n_episodes_approx": 110_000,
        "dataset_cls": None,
        "data_type": "vla",
    },
    "taco-play": {
        "hf_path": "lerobot/taco_play",
        "description": "TACO-Play — 3.2K kitchen manipulation demos (RGB + depth + end-effector).",
        "paper_reference": "Rosete-Beas et al. (2023) TACO-Play",
        "n_episodes_approx": 3_200,
        "data_type": "vla",
    },
    "pusht": {
        "hf_path": "lerobot/pusht",
        "description": "Push-T — 300 2D pushing demos for diffusion policy benchmarking.",
        "paper_reference": "Chi et al. (2023) Diffusion Policy",
        "n_episodes_approx": 300,
        "data_type": "vla",
    },
    "aloha-sim": {
        "hf_path": "lerobot/aloha_sim_insertion_human",
        "description": "ALOHA simulation — 1K bimanual insertion episodes (sim).",
        "paper_reference": "Zhao et al. (2023) ACT",
        "n_episodes_approx": 1_000,
        "data_type": "vla",
    },
    # -----------------------------------------------------------------------
    # Embodied-oriented VLM data (spatial / affordance, no action labels)
    # Key reference: EmbodiedMidtrain (2026) — proximity-based data engine
    # -----------------------------------------------------------------------
    "refspatial": {
        "hf_path": "RefSpatial/RefSpatial",
        "subsets": None,
        "description": (
            "RefSpatial — spatial referring and reasoning dataset for embodied agents. "
            "Bridges the VLM→VLA gap by training on spatial grounding without action labels."
        ),
        "paper_reference": "RefSpatial (2025)",
        "n_episodes_approx": 100_000,
        "data_type": "vlm",
    },
    "embspatial-bench": {
        "hf_path": "EmbSpatial/EmbSpatial-Bench",
        "subsets": None,
        "description": (
            "EmbSpatial-Bench — embodied spatial understanding VQA. "
            "Tests relative positions, distances, directions in 3D scene context."
        ),
        "paper_reference": "EmbSpatial (2024)",
        "n_episodes_approx": 10_000,
        "data_type": "vlm",
    },
    "robo2vlm": {
        "hf_path": "Robo2VLM/Robo2VLM",
        "subsets": None,
        "description": (
            "Robo2VLM — robotic visual question answering generated from robot "
            "observation trajectories. No action labels; aligns VLM priors to "
            "robot-camera viewpoints and manipulation contexts."
        ),
        "paper_reference": "Robo2VLM (2025)",
        "n_episodes_approx": 500_000,
        "data_type": "vlm",
    },
    "robopoint": {
        "hf_path": "wentao-yuan/robopoint-data",
        "subsets": None,
        "description": (
            "RoboPoint — spatial affordance prediction: given an image + instruction, "
            "predict the target 2D point for manipulation. Trains spatial reasoning "
            "without requiring low-level action labels."
        ),
        "paper_reference": "Yuan et al. (2024) RoboPoint",
        "n_episodes_approx": 600_000,
        "data_type": "vlm",
    },
    "vln-r2r": {
        "hf_path": "prs-eth/room_across_the_room",
        "subsets": None,
        "description": (
            "R2R (Room-to-Room) — vision-and-language navigation trajectories "
            "in photorealistic indoor environments. Trajectory-centric supervision "
            "for spatial grounding and instruction following."
        ),
        "paper_reference": "Anderson et al. (2018) R2R",
        "n_episodes_approx": 22_000,
        "data_type": "vlm",
    },
}


class MidTrainDatasetRegistry:
    """Factory that assembles a mid-training DataLoader from registered datasets."""

    def __init__(self, **config_overrides) -> None:
        from .mid_train_dataset_config import MidTrainDatasetConfig  # lazy: avoids an import cycle
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
            raise DatasetError(
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
