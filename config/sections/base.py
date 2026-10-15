"""Shared machinery of the Halo-VLA run-config sections.

Every section knows how its fields reach the training script: as a ``scripts/train.py``
argument (``ARGS``: field -> argparse dest) or as a ``HaloVLMConfig`` field the CLI does not
expose (``OVERRIDES``: field -> dataclass field). Only fields a YAML file actually sets are
forwarded, so the CLI's and the dataclass's own defaults stay the single source of truth and
command-line flags still override the YAML.
"""

from __future__ import annotations

from typing import Any, ClassVar

from hale_core.config.sections.common import StrictModel


class VLASection(StrictModel):
    """A run-config section that maps its explicitly set fields onto train.py / HaloVLMConfig."""

    #: field -> ``scripts/train.py`` argparse dest.
    ARGS: ClassVar[dict[str, str]] = {}
    #: field -> ``HaloVLMConfig`` field (architecture settings without a CLI flag).
    OVERRIDES: ClassVar[dict[str, str]] = {}

    def explicitly_set(self) -> dict[str, Any]:
        """Fields given in the YAML (or constructor), excluding nested sections."""
        return {k: getattr(self, k) for k in self.model_fields_set if not isinstance(getattr(self, k), VLASection)}

    def train_arguments(self) -> dict[str, Any]:
        """``{argparse dest: value}`` for this section and its nested sections."""
        out = {self.ARGS[k]: v for k, v in self.explicitly_set().items() if k in self.ARGS}
        for child in self._children():
            out.update(child.train_arguments())
        return out

    def model_overrides(self) -> dict[str, Any]:
        """``{HaloVLMConfig field: value}`` for this section and its nested sections."""
        out = {self.OVERRIDES[k]: v for k, v in self.explicitly_set().items() if k in self.OVERRIDES}
        for child in self._children():
            out.update(child.model_overrides())
        return out

    def _children(self) -> list[VLASection]:
        return [v for k in type(self).model_fields if isinstance(v := getattr(self, k), VLASection)]
