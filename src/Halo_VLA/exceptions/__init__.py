"""Halo-VLA exception hierarchy (import as ``exceptions``; ``src/Halo_VLA`` is an import root).

.. code-block:: text

    HaleCoreError                         (hale_core.exceptions)
    └── HaloVLAError
        ├── HaloVLAConfigError            (also hale_core ConfigError, ValueError)
        ├── ModelInputError               (also ValueError)
        ├── ModelStateError               (also RuntimeError)
        ├── DatasetError                  (also RuntimeError)
        ├── DatasetNotFoundError          (also FileNotFoundError)
        ├── CheckpointNotFoundError       (also FileNotFoundError)
        ├── EvaluationError               (also ValueError)
        └── TrainingEntryError            (also ImportError)

Each class also derives from the built-in exception it replaced, so code and tests written
against ``ValueError``/``RuntimeError``/``FileNotFoundError`` keep working. Missing optional
packages raise :class:`hale_core.exceptions.MissingDependencyError` (an ``ImportError``).
"""

from exceptions.checkpoint_not_found_error import CheckpointNotFoundError
from exceptions.dataset_error import DatasetError
from exceptions.dataset_not_found_error import DatasetNotFoundError
from exceptions.evaluation_error import EvaluationError
from exceptions.halo_vla_config_error import HaloVLAConfigError
from exceptions.halo_vla_error import HaloVLAError
from exceptions.model_input_error import ModelInputError
from exceptions.model_state_error import ModelStateError
from exceptions.training_entry_error import TrainingEntryError

__all__ = [
    "CheckpointNotFoundError",
    "DatasetError",
    "DatasetNotFoundError",
    "EvaluationError",
    "HaloVLAConfigError",
    "HaloVLAError",
    "ModelInputError",
    "ModelStateError",
    "TrainingEntryError",
]
