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
        )"""

from .pretrain_dataset_config import PRETRAIN_DATASET_NAMES, PretrainDatasetConfig  # noqa: F401
from .pretrain_dataset_registry import (  # noqa: F401
    PRETRAIN_DATASETS,
    PretrainDatasetRegistry,
    _HFRobotDataset,
)

__all__ = [
    "PRETRAIN_DATASETS",
    "PRETRAIN_DATASET_NAMES",
    "PretrainDatasetConfig",
    "PretrainDatasetRegistry",
]
