"""The Halo-VLA run config: one YAML file -> ``scripts/train.py`` arguments + ``HaloVLMConfig``.

Same layout and loader as Hale-VLM (HaleBlocks' registered pydantic schemas, ``inherits:`` for
YAML inheritance)::

    variant: halo_vla_qwen3_5
    model:
      vision: {encoder: siglip, projector_type: qformer, qformer: {num_queries: 32}}
      llm: {backbone: qwen3.5-4b-base, use_lora: true, lora_r: 16}
      dual_system: {preset: openhelix, async_steps: 60}
    train: {epochs: 10, batch_size: 8, lr: 1.0e-4}
    data: {dataset: droid, droid: {data_root: /data/droid_100}}

``python scripts/train.py --config configs/base.yaml [--any_flag value ...]``: the YAML provides
the defaults, flags given on the command line still win. Only keys present in the YAML are
forwarded; everything else keeps the script's / ``HaloVLMConfig``'s own default. Needs
HaleBlocks (``pip install -e '.[config]'``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from exceptions import HaloVLAConfigError

from hale_core.config import load_registered_config
from hale_core.config.sections.common import StrictModel
from hale_core.registry import register_config
from pydantic import Field

from config.sections.base import VLASection
from config.sections import (
    VLADataConfig,
    VLAEvalConfig,
    VLALoggingConfig,
    VLAModelConfig,
    VLATrainConfig,
    VLAVizConfig,
)


@register_config("vla")
class VLARunConfig(StrictModel):
    """Full configuration of one Halo-VLA training run."""

    variant: str = "halo_vla"
    device: str | None = None              # cpu / cuda / cuda:1 / mps; None = cuda if available
    model: VLAModelConfig = Field(default_factory=VLAModelConfig)
    train: VLATrainConfig = Field(default_factory=VLATrainConfig)
    data: VLADataConfig = Field(default_factory=VLADataConfig)
    eval: VLAEvalConfig = Field(default_factory=VLAEvalConfig)
    logging: VLALoggingConfig = Field(default_factory=VLALoggingConfig)
    viz: VLAVizConfig = Field(default_factory=VLAVizConfig)

    def sections(self) -> list[VLASection]:
        """The nested sections that map onto ``scripts/train.py`` arguments."""
        return [self.model, self.train, self.data, self.eval, self.logging, self.viz]

    def train_arguments(self) -> dict[str, Any]:
        """``{scripts/train.py argparse dest: value}`` for every key the YAML sets."""
        out: dict[str, Any] = {}
        for section in self.sections():
            out.update(section.train_arguments())
        if "device" in self.model_fields_set and self.device is not None:
            out["device"] = self.device
        return out

    def model_overrides(self) -> dict[str, Any]:
        """``{HaloVLMConfig field: value}`` for architecture keys the CLI has no flag for."""
        out: dict[str, Any] = {}
        for section in self.sections():
            out.update(section.model_overrides())
        return out


def load_vla_config(path: str | Path) -> VLARunConfig:
    """Load ``path`` (following ``inherits:``) and validate it as a :class:`VLARunConfig`."""
    cfg = load_registered_config(path, schema="vla")
    if not isinstance(cfg, VLARunConfig):
        raise HaloVLAConfigError(f"{path} did not validate as a Halo-VLA run config (got {type(cfg).__name__})")
    return cfg


def save_resolved_config(source: str | Path, destination: str | Path, argv: list[str] | None = None) -> Path:
    """Write ``source`` with its inheritance resolved to ``destination``.

    Only keys set somewhere in the ``inherits:`` chain are written (unset keys keep the code's
    defaults, and an unset dual-system option must stay unset rather than become ``null``).
    ``argv`` (the command line) is stored as a comment: flags given there override the YAML.
    """
    cfg = load_vla_config(source)
    header = f"# resolved from {Path(source).resolve()}\n"
    if argv:
        header += f"# command-line overrides: {' '.join(argv)}\n"
    destination = Path(destination)
    destination.write_text(header + yaml.safe_dump(cfg.model_dump(mode="json", exclude_unset=True), sort_keys=False))
    return destination
