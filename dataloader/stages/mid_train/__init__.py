"""Phase 2 — Mid-training dataloaders for Halo-VLA.

Mid-training uses curated domain-specific data to adapt pretrained representations
to the target workspace, object distribution, and instruction-following style.
This phase is also called "embodied alignment" or "VLM-to-VLA bridging".

VLM datasets (vision-language connector pretraining):
  - ``llava-pretrain-558k``    : LLaVA Pretrain 558K curated BLIP captions
  - ``sharegpt4v-pt``          : ShareGPT4V-PT 1.2M GPT-4V captions
  - ``blip-laion-cc-sbu-558k`` : BLIP LAION+CC+SBU 558K captions
  - ``recap-datacomp-1b``      : Recap DataComp-1B 1.28B recaptioned pairs
  - ``allava-vflan``           : AllaVA-vFLAN 1.3M task-diverse QA

Embodied-oriented VLM data (spatial / affordance, no action labels):
  - ``refspatial``             : RefSpatial — spatial referring and reasoning
  - ``embspatial-bench``       : EmbSpatial-Bench — embodied spatial understanding VQA
  - ``robo2vlm``               : Robo2VLM — robotic VQA from robot observations
  - ``robopoint``              : RoboPoint — spatial affordance prediction data
  - ``vln-r2r``                : R2R — vision-and-language navigation trajectories

VLA datasets (robot domain adaptation):
  - ``droid``       : DROID full Franka corpus (76K diverse demos)
  - ``airoa-moma``  : AIRoA-MoMA HSR teleoperation (RGB video + metadata)
  - ``eo-data-qa``  : EO-Data1.5M QA subsets (12 task-structured categories)
  - ``rh20t``       : RH20T contact-rich manipulation (110K demos)
  - ``taco-play``   : TACO-Play kitchen manipulation (3.2K demos)
  - ``pusht``       : Push-T 2D pushing (300 demos)
  - ``aloha-sim``   : ALOHA simulation bimanual (1K demos)

Usage::

    from dataloader.stages.mid_train import MidTrainDatasetRegistry

    registry = MidTrainDatasetRegistry(batch_size=64)
    loader = registry.build()"""

from .mid_train_dataset_config import MID_TRAIN_DATASET_NAMES, MidTrainDatasetConfig  # noqa: F401
from .mid_train_dataset_registry import (  # noqa: F401
    MID_TRAIN_DATASETS,
    MidTrainDatasetRegistry,
    _HFRobotDataset,
)

__all__ = [
    "MID_TRAIN_DATASETS",
    "MID_TRAIN_DATASET_NAMES",
    "MidTrainDatasetConfig",
    "MidTrainDatasetRegistry",
]
