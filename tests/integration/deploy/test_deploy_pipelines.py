"""End-to-end exports of tiny Halo-VLA policies with numerical parity (ONNX Runtime, MLX)."""

from __future__ import annotations

import json

import pytest
from hale_core.deploy import DeployRunner

from tests.support.deploy_support import CONFIGS

pytestmark = pytest.mark.integration

ONNX = sorted(CONFIGS.glob("ci_*_onnx_ort.yaml"))
MLX = sorted(CONFIGS.glob("ci_*_mlx_fp32.yaml"))


def run_and_read_parity(path, tmp_path) -> dict:
    report = DeployRunner.from_file(path, work_dir=tmp_path).run()
    assert report.ok, [(r.name, r.status, r.message) for r in report.results]
    parity = next(r for r in report.results if r.name.endswith("parity"))
    return json.loads(parity.artifacts[0].path.read_text())


@pytest.mark.parametrize("path", ONNX, ids=lambda p: p.stem)
def test_onnx_runtime_parity_at_every_shape_corner(path, tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    parity = run_and_read_parity(path, tmp_path)
    assert set(parity) == {"min", "opt", "max"}
    for corner in parity.values():
        for name, metrics in corner.items():
            assert metrics["ok"] and metrics["max_abs"] < 1e-4, (name, metrics)


@pytest.mark.parametrize("path", MLX, ids=lambda p: p.stem)
def test_mlx_parity(path, tmp_path) -> None:
    pytest.importorskip("mlx.core")
    parity = run_and_read_parity(path, tmp_path)
    for corner in parity.values():
        for name, metrics in corner.items():
            assert metrics["ok"] and metrics["max_abs"] < 1e-4, (name, metrics)


def test_exported_perception_package_records_the_prompt_contract(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    report = DeployRunner.from_file(
        CONFIGS / "ci_perception_onnx_ort.yaml", work_dir=tmp_path
    ).run()
    onnx = next(a for r in report.results for a in r.artifacts if a.kind == "onnx")
    assert onnx.metadata["inputs"] == ["images", "input_ids", "attention_mask", "states"]
    assert onnx.metadata["outputs"] == ["action_context"]
