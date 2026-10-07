"""Phase 1 — Pre-training dataloaders for Halo-VLA.

Pre-training uses web-scale video/image data and large open-world robot
teleoperation corpora to learn generalised visual-motor representations.

VLM datasets (image-text alignment):
  - ``laion-aesthetics``       : LAION-2B-en aesthetic subset (≥5.0 quality)
  - ``cc3m``                   : Conceptual Captions 3M
  - ``cc12m``                  : Conceptual Captions 12M
  - ``datacomp-1b``            : DataComp-1B CommonPool
  - ``wit``                    : Wikipedia Image Text
  - ``redcaps``                : RedCaps 12M Reddit captions
  - ``something-something-v2`` : SSv2 — 220K procedural human-object video clips

VLA datasets (robot teleoperation):
  - ``eo-data-interleave``  : EO-Data1.5M interleaved subsets (web robot QA)
  - ``open-x-pretrain``     : Open X-Embodiment large-scale mix
  - ``bridge-v2``           : Bridge Data V2 (60K diverse manipulation demos)
  - ``fractal-rt1``         : Google RT-1 training corpus (130K episodes)
  - ``bc-z``                : BC-Z 25K episodes across 100 tasks

Synthetic/simulation datasets:
  - ``syngrasp-1b``         : SynGrasp-1B — 1B randomised grasp scenes (GraspVLA)
  - ``robocasa``            : RoboCasa — scalable household manipulation rollouts

Egocentric human video (cross-embodiment bridge):
  - ``ego4d``               : Ego4D — 3,600 h first-person video from 74 scenarios
  - ``vitra``               : VITRA — in-the-wild hand videos → (img, instr, action)
  - ``egovla``              : EgoVLA — large-scale egocentric video for VLA pretraining

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
    # -----------------------------------------------------------------------
    # VLM image-text alignment datasets
    # -----------------------------------------------------------------------
    "laion-aesthetics": {
        "hf_path": "laion/laion2B-en-aesthetic",
        "subsets": None,
        "description": "LAION-2B-en aesthetic subset (≥5.0); ~600M image-text pairs.",
        "paper_reference": "Schuhmann et al. (2022) LAION-5B",
        "n_episodes_approx": 600_000_000,
        "data_type": "vlm",
    },
    "cc3m": {
        "hf_path": "pixparse/cc3m-wds",
        "subsets": None,
        "description": "Conceptual Captions 3M — 3.3M image-alt-text pairs.",
        "paper_reference": "Sharma et al. (2018) CC3M",
        "n_episodes_approx": 3_300_000,
        "data_type": "vlm",
    },
    "cc12m": {
        "hf_path": "pixparse/cc12m-wds",
        "subsets": None,
        "description": "Conceptual Captions 12M — ~12M image-text pairs.",
        "paper_reference": "Changpinyo et al. (2021) CC12M",
        "n_episodes_approx": 12_000_000,
        "data_type": "vlm",
    },
    "datacomp-1b": {
        "hf_path": "mlfoundations/datacomp_1b",
        "subsets": None,
        "description": "DataComp-1B CommonPool — 1.28B CLIP-filtered image-text pairs.",
        "paper_reference": "Gadre et al. (2023) DataComp",
        "n_episodes_approx": 1_280_000_000,
        "data_type": "vlm",
    },
    "wit": {
        "hf_path": "google/wit",
        "subsets": None,
        "description": "Wikipedia-based Image Text — 37.6M curated image-caption pairs.",
        "paper_reference": "Srinivasan et al. (2021) WIT",
        "n_episodes_approx": 37_600_000,
        "data_type": "vlm",
    },
    "redcaps": {
        "hf_path": "red_caps",
        "subsets": None,
        "description": "RedCaps — 12M human-written image-text pairs from Reddit.",
        "paper_reference": "Desai et al. (2021) RedCaps",
        "n_episodes_approx": 12_000_000,
        "data_type": "vlm",
    },
    # -----------------------------------------------------------------------
    # VLA robot teleoperation datasets
    # -----------------------------------------------------------------------
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
        "data_type": "vla",
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
        "data_type": "vla",
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
        "data_type": "vla",
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
        "data_type": "vla",
    },
    "bc-z": {
        "hf_path": "lerobot/bc_z",
        "subsets": None,
        "description": "BC-Z — 25K robot episodes across 100 tasks.",
        "paper_reference": "Jang et al. (2022) BC-Z",
        "n_episodes_approx": 25_000,
        "data_type": "vla",
    },
    "droid-v1": {
        "hf_path": "lerobot/droid_1.0.1",
        "subsets": None,
        "description": (
            "DROID 1.0.1 — 76K trajectories, 564 scenes, 86 tasks, 50 operators. "
            "In-the-wild Franka Panda; high scene diversity; key OXE complement."
        ),
        "paper_reference": "Khazatsky et al. (2024) DROID",
        "n_episodes_approx": 76_000,
        "data_type": "vla",
    },
    "libero-pretrain": {
        "hf_path": "HuggingFaceVLA/libero",
        "subsets": None,
        "description": (
            "LIBERO (HuggingFaceVLA edition) — 130+ tasks, 5K+ episodes on Franka. "
            "LeRobot v0.4.0 official. Multi-task instruction-following pretraining."
        ),
        "paper_reference": "Liu et al. (2023) LIBERO",
        "n_episodes_approx": 5_000,
        "data_type": "vla",
    },
    "openEAI-dataset": {
        "hf_path": "OpenEAI/OpenEAI-Dataset",
        "subsets": None,
        "description": (
            "OpenEAI-Dataset — ~3.12 TB unified HDF5 aggregating "
            "OXE + UMI Community + DROID + BC-Z in a single format."
        ),
        "paper_reference": "OpenEAI (2024)",
        "n_episodes_approx": 3_000_000,
        "data_type": "vla",
    },
    "lerobot-community-v3": {
        "hf_path": "lerobot/community_dataset_v3",
        "subsets": None,
        "description": (
            "LeRobot Community Dataset v3 — 791 datasets across 46 robot types, "
            "from 851 sources. Datasets v3.0 format, OXE-scale chunked episodes."
        ),
        "paper_reference": "Cadène et al. (2024) LeRobot",
        "n_episodes_approx": 5_000_000,
        "data_type": "vla",
    },
    "robogene": {
        "hf_path": "X-Humanoid/RoboGene",
        "subsets": None,
        "description": (
            "RoboGene — diversity-driven agentic VLA pretraining. "
            "Addresses limited scene variety and physical grounding. LeRobot-compatible."
        ),
        "paper_reference": "X-Humanoid (2024) RoboGene",
        "n_episodes_approx": 500_000,
        "data_type": "vla",
    },
    "being-h0": {
        "hf_path": "BeingBeyond/Being-H0",
        "subsets": None,
        "description": (
            "Being-H0 — large-scale human video pretraining via explicit hand motion "
            "modelling. Bridges embodiment gap with egocentric 2D/3D cues."
        ),
        "paper_reference": "BeingBeyond (2024) Being-H0",
        "n_episodes_approx": 100_000,
        "data_type": "vla",
    },
    "agibot-world": {
        "hf_path": "lerobot/xvla-agibot-world",
        "subsets": None,
        "description": (
            "AgiBot World — bimanual real-world manipulation used in X-VLA pretraining. "
            "Dexterous tasks with rich scene diversity."
        ),
        "paper_reference": "AgiBot (2024) AgiBot World",
        "n_episodes_approx": 200_000,
        "data_type": "vla",
    },
    "h-tac-ttp": {
        "hf_path": "BeingBeyond/TTP",
        "subsets": None,
        "description": (
            "H-Tac TTP (Tactile Transformer Pretraining) — tactile sensor "
            "pretraining for dexterous manipulation. Contact dynamics beyond vision."
        ),
        "paper_reference": "BeingBeyond (2024) H-Tac",
        "n_episodes_approx": 50_000,
        "data_type": "vla",
    },
    # -----------------------------------------------------------------------
    # Synthetic / simulation corpora
    # -----------------------------------------------------------------------
    "syngrasp-1b": {
        "hf_path": "GraspVLA/SynGrasp-1B",
        "subsets": None,
        "description": (
            "SynGrasp-1B — 1 billion procedurally-generated grasp scenes with "
            "randomised objects, lighting, and camera poses (GraspVLA). "
            "Provides robust geometric pretraining at scale."
        ),
        "paper_reference": "GraspVLA (2025) SynGrasp-1B",
        "n_episodes_approx": 1_000_000_000,
        "data_type": "vla",
    },
    "robocasa": {
        "hf_path": "lerobot/robocasa",
        "subsets": None,
        "description": (
            "RoboCasa — scalable household manipulation rollouts across diverse "
            "kitchen/living layouts. Enables generalist policy pretraining."
        ),
        "paper_reference": "Nasiriany et al. (2024) RoboCasa",
        "n_episodes_approx": 100_000,
        "data_type": "vla",
    },
    # -----------------------------------------------------------------------
    # Egocentric human video (cross-embodiment bridge)
    # -----------------------------------------------------------------------
    "ego4d": {
        "hf_path": "facebook/ego4d",
        "subsets": None,
        "description": (
            "Ego4D — 3,600 hours of first-person video from 931 participants "
            "across 74 worldwide scenarios. Low-cost VLA pretraining source."
        ),
        "paper_reference": "Grauman et al. (2022) Ego4D",
        "n_episodes_approx": 9_600,
        "data_type": "vla",
        "trust_remote_code": True,
    },
    "vitra": {
        "hf_path": "VITRA-Dataset/VITRA",
        "subsets": None,
        "description": (
            "VITRA (ICRA 2026) — converts in-the-wild human hand videos into "
            "(image, instruction, action) tuples for VLA pretraining. "
            "Bridges embodiment gap via egocentric hand motion."
        ),
        "paper_reference": "VITRA (2026) ICRA",
        "n_episodes_approx": 500_000,
        "data_type": "vla",
    },
    "egovla": {
        "hf_path": "EgoVLA/EgoVLA",
        "subsets": None,
        "description": (
            "EgoVLA — large-scale egocentric human video corpus for VLA pretraining. "
            "Overcomes robot data scarcity via human-to-robot cross-embodiment transfer."
        ),
        "paper_reference": "EgoVLA (2025)",
        "n_episodes_approx": 1_000_000,
        "data_type": "vla",
    },
    # -----------------------------------------------------------------------
    # Language-vision co-training (backbone preservation)
    # -----------------------------------------------------------------------
    "something-something-v2": {
        "hf_path": "HuggingFaceM4/something-something-v2",
        "subsets": None,
        "description": (
            "Something-Something V2 — 220K crowd-sourced procedural video clips "
            "showing humans performing fine-grained actions with everyday objects. "
            "VL co-training for spatial and temporal reasoning preservation."
        ),
        "paper_reference": "Goyal et al. (2017) SSv2",
        "n_episodes_approx": 220_847,
        "data_type": "vlm",
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
