"""A YAML run config drives ``scripts/train.py`` end to end (inheritance -> typed config -> training)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("hale_core")

from config.run_config import load_vla_config  # noqa: E402
from models.halo_vla import HaloVLM  # noqa: E402
from tests.support.training_support import load_train_script  # noqa: E402
from training.train_datasets import TRAIN_DATASETS  # noqa: E402

CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def script():
    return load_train_script()


def test_train_runs_a_dual_system_from_yaml(script, tmp_path: Path, monkeypatch) -> None:
    """A YAML that inherits overfit.yaml (-> base.yaml) drives a real HaloVLM + OpenHelix run."""
    script.HaloVLM = HaloVLM

    class Rows(list):
        tokenizer = type(
            "Tok", (), {"pad_token_id": 0, "decode": staticmethod(lambda *a, **k: "")}
        )()

    def build_loader(img_size: int, **_kw):
        g = torch.Generator().manual_seed(0)
        rows = Rows(
            {
                "images": torch.rand(2, 3, img_size, img_size, generator=g),
                "input_ids": torch.randint(1, 50, (8,), generator=g),
                "attention_mask": torch.ones(8, dtype=torch.long),
                "labels": torch.randint(1, 50, (8,), generator=g),
                "actions": torch.randn(4, 4, generator=g),
                "action_mask": torch.ones(4),
                "states": torch.randn(1, 3, generator=g),
            }
            for _ in range(8)
        )
        return DataLoader(rows, batch_size=2)

    monkeypatch.setitem(
        TRAIN_DATASETS.items, "eo", lambda args, config, tokenizer: build_loader(config.img_size)
    )
    run = tmp_path / "run.yaml"
    run.write_text(f"""inherits: {CONFIGS / "overfit.yaml"}
variant: tiny_openhelix
device: cpu
model:
  emb_dim: 32
  decoder: {{num_layers: 1, num_heads: 4, mlp_dim: 32}}
  vision: {{image_size: 32, vit: {{num_layers: 1, num_heads: 4, mlp_dim: 32}}}}
  action: {{action_dim: 4, state_dim: 3, chunk_size: 2, flow: {{hidden_dim: 16}}}}
  dual_system: {{preset: openhelix, prealign_steps: 2, system1_dim: 32, system1_heads: 4, system1_layers: 1}}
train: {{epochs: 2, checkpoint: {{dir: {tmp_path}, every_n_steps: 1, keep_last: 0}}}}
data: {{max_samples: 8}}
eval: {{val_fraction: 0.25}}
""")
    args = script.parse_args(["--config", str(run)])
    script.train(args)
    ckpt = torch.load(tmp_path / "halo_vla_epoch2.pt", weights_only=False)
    config = ckpt["config"]
    assert (config.emb_dim, config.img_size, config.use_moe, config.dual_system) == (
        32,
        32,
        False,
        "openhelix",
    )
    assert config.dual_system_options["prealign_steps"] == 2
    assert any(k.startswith("system1.") for k in ckpt["model_state_dict"])
    assert f"# command-line overrides: --config {run}" in (tmp_path / "run_config.yaml").read_text()
    saved = load_vla_config(
        tmp_path / "run_config.yaml"
    )  # provenance: resolved YAML next to the checkpoints
    assert saved.variant == "tiny_openhelix" and saved.model.dual_system.preset == "openhelix"
    assert saved.model.world_model.enabled is False  # inherited from overfit.yaml
