"""Every config under configs/deploy builds into a valid, executable-in-order pipeline."""

from __future__ import annotations

import pytest
from tests.support.deploy_support import CONFIGS

from hale_core.deploy import DeployRunner
from hale_core.deploy.config import Config

TARGETS = sorted(p for p in CONFIGS.glob("*.yaml"))


def test_there_are_targets_for_every_vendor() -> None:
    vendors = {Config.fromfile(p).get_path("device.vendor") for p in TARGETS}
    assert vendors == {"nvidia", "apple", "amd", "qualcomm", None}  # None: vendor-neutral ONNX Runtime CI configs
    assert len(TARGETS) >= 30


@pytest.mark.parametrize("path", TARGETS, ids=lambda p: p.stem)
def test_config_builds_a_valid_pipeline(path, tmp_path) -> None:
    runner = DeployRunner.from_file(path, work_dir=tmp_path)
    pipeline, ctx = runner.build()
    pipeline.validate()
    assert runner.spec.custom_imports == ["Halo_VLA.deploy"]
    assert runner.spec.adapter["type"].startswith("halo_vla.")
    assert runner.spec.name == path.stem
    if not path.stem.endswith("_onnx_ort"):  # ONNX Runtime CI configs are vendor-neutral
        assert ctx.device["vendor"] in {"nvidia", "apple", "amd", "qualcomm"}
