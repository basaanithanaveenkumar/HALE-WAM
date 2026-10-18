"""The Halo-VLA exception hierarchy is rooted in hale_core and keeps the built-in bases it replaced."""

from __future__ import annotations

import pytest
from hale_core.exceptions import ConfigError, HaleCoreError

from exceptions import (
    CheckpointNotFoundError,
    DatasetError,
    DatasetNotFoundError,
    EvaluationError,
    HaloVLAConfigError,
    HaloVLAError,
    ModelInputError,
    ModelStateError,
    TrainingEntryError,
)


@pytest.mark.parametrize(
    "error, builtin",
    [
        (HaloVLAConfigError, ValueError),
        (ModelInputError, ValueError),
        (ModelStateError, RuntimeError),
        (DatasetError, RuntimeError),
        (DatasetNotFoundError, FileNotFoundError),
        (CheckpointNotFoundError, FileNotFoundError),
        (EvaluationError, ValueError),
        (TrainingEntryError, ImportError),
    ],
)
def test_every_error_is_a_halo_vla_error_and_its_old_builtin(error: type, builtin: type) -> None:
    assert (
        issubclass(error, HaloVLAError)
        and issubclass(error, HaleCoreError)
        and issubclass(error, builtin)
    )


def test_config_errors_are_also_hale_core_config_errors() -> None:
    assert issubclass(HaloVLAConfigError, ConfigError)
    with pytest.raises(ValueError, match="bad"):
        raise HaloVLAConfigError("bad value")
