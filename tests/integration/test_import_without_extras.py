"""Every package module imports with the optional extras missing (they are imported lazily)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Optional extras (see ``[project.optional-dependencies]``) and heavy third-party libraries.
BLOCKED = (
    "ray",
    "webdataset",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "peft",
    "transformers",
    "cv2",
    "imageio",
    "matplotlib",
    "PIL",
    "tensorboard",
    "safetensors",
    "mlx",
    "tensorrt",
    "qai_hub",
    "lifting",
    "datasets",
    "torchcodec",
    "decord",
    "av",
)

SCRIPT = textwrap.dedent(
    """
    import importlib, pkgutil, sys

    BLOCKED = set({blocked!r})

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError("optional dependency not installed: " + name)
            return None

    sys.meta_path.insert(0, Blocker())
    sys.argv = sys.argv[:1]
    skip_prefixes = ("tests", "scripts", "ext_pkgs", "deploy", "notebooks")
    failed = []
    for root in {roots!r}:
        for info in pkgutil.walk_packages([root], prefix=""):
            name = info.name
            if name.startswith(skip_prefixes) or name.endswith("__main__"):
                continue
            try:
                importlib.import_module(name)
            except Exception as exc:  # report every failure, not only the first
                failed.append(f"{{name}}: {{type(exc).__name__}}: {{str(exc)[:120]}}")
    importlib.import_module("Halo_VLA.deploy")
    print("\\n".join(failed))
    sys.exit(1 if failed else 0)
    """
)


def test_all_modules_import_without_optional_extras() -> None:
    pytest.importorskip("hale_core")
    # ``src/Halo_VLA`` holds models / dual_system / training ...; the repo root holds config / dataloader / utils.
    roots = [str(ROOT / "src" / "Halo_VLA"), str(ROOT)]
    code = SCRIPT.format(blocked=BLOCKED, roots=roots)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
    }  # the parent's import roots
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
