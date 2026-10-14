"""Training runtime for Halo-VLA: DDP-safe training step, distributed session, Ray and WebDataset.

* :class:`HaloVLATrainingStep` - model forward + language, action-flow and visual losses in one
  ``nn.Module.forward`` (what DistributedDataParallel needs to track every parameter).
* :class:`DDPSession` - rank / device / process group for single-process, torchrun or Ray runs;
  wraps the step in DDP and gives every rank its own slice of the data.
* :class:`RayCliOptions`, :class:`WebDatasetCliOptions` - ``--ray*`` / ``--webdataset*`` flags of
  ``scripts/train.py`` mapped onto HaleBlocks' ``RayConfig`` / ``WebDatasetConfig``.
* :func:`launch_train_on_ray` - run ``train(args)`` on Ray Train workers.
* :class:`VLAEvaluator`, :func:`split_validation` - held-out losses and action metrics.

Ray and WebDataset come from HaleBlocks and are optional (``pip install -e '.[distributed]'``);
nothing here imports them unless the corresponding flag is used.
"""

from training.ddp_session import DDPSession
from training.dual_system_step import DualSystemTrainingStep
from training.evaluator import VLAEvaluator
from training.ray_cli import RayCliOptions
from training.ray_launch import launch_train_on_ray
from training.training_step import HaloVLATrainingStep, StepOutput
from training.validation_split import split_validation
from training.webdataset_cli import WebDatasetCliOptions

__all__ = [
    "DDPSession",
    "DualSystemTrainingStep",
    "HaloVLATrainingStep",
    "RayCliOptions",
    "StepOutput",
    "VLAEvaluator",
    "WebDatasetCliOptions",
    "launch_train_on_ray",
    "split_validation",
]
