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

VLA datasets (robot task fine-tuning / SFT):
  - ``droid-task``             : DROID task-specific subset (curated per task)
  - ``eo-data-task``           : EO-Data1.5M task-targeted subsets
  - ``airoa-task``             : AIRoA-MoMA task-filtered episodes
  - ``libero-goal``            : LIBERO-Goal 10-task goal-conditioned benchmark
  - ``libero-spatial``         : LIBERO-Spatial 10-task spatial reasoning
  - ``libero-object``          : LIBERO-Object 10-task object-manipulation
  - ``libero-100``             : LIBERO-100 full 100-task benchmark
  - ``aloha-bimanual``         : ALOHA real bimanual demos (200 eps)
  - ``metaworld-mt50``         : Meta-World MT50 (50 tasks, 2,500 eps)

Preference / DPO-style datasets (winner/loser trajectory pairs):
  - ``flowpro-pairs``          : FlowPRO rollback preference pairs (local collection)
  - ``apo-interventions``      : APO human-in-the-loop correction preferences (local)

Offline RL / hindsight relabeling:
  - ``hindsight-relabeled``    : LfH hindsight relabeled failed rollouts (local)

DAgger / interactive imitation:
  - ``dagger-corrections``     : Human gated-takeover correction trajectories (local)

Usage::

    from dataloader.stages.post_train import PostTrainDatasetRegistry

    registry = PostTrainDatasetRegistry(
        task_filter="pick_and_place",
        batch_size=32,
    )
    loader = registry.build()"""

from .post_train_dataset_config import (  # noqa: F401
    POST_TRAIN_DATASET_NAMES,
    PostTrainDatasetConfig,
)
from .post_train_dataset_registry import (  # noqa: F401
    POST_TRAIN_DATASETS,
    PostTrainDatasetRegistry,
    _HFRobotDataset,
    _LocalRobotDataset,
)

__all__ = [
    "POST_TRAIN_DATASETS",
    "POST_TRAIN_DATASET_NAMES",
    "PostTrainDatasetConfig",
    "PostTrainDatasetRegistry",
]
