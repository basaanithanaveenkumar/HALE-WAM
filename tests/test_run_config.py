"""Hale-VLM-style YAML run configs for Halo-VLA (config/run_config.py, scripts/train.py --config)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src" / "Halo_VLA"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("hale_core")
from pydantic import ValidationError  # noqa: E402

from config import HaloVLMConfig  # noqa: E402
from config.overrides import apply_model_overrides, parse_model_options  # noqa: E402
from config.run_config import VLARunConfig, load_vla_config  # noqa: E402
from config.sections.base import VLASection  # noqa: E402
from config.sections.dual_system import DualSystemSection  # noqa: E402
from dual_system.config import DualSystemConfig, parse_overrides, resolve_dual_system_config  # noqa: E402

from tests.training_support import load_train_script  # noqa: E402

CONFIGS = ROOT / "configs"
RUN_CONFIGS = sorted(p for p in CONFIGS.rglob("*.yaml") if "roadmap" not in p.parts)
META = ("config", "model_overrides", "run_variant", "argv")


@pytest.fixture(scope="module")
def script():
    return load_train_script()


def sections(section: VLASection):
    yield section
    for name in type(section).model_fields:
        child = getattr(section, name)
        if isinstance(child, VLASection):
            yield from sections(child)


# ---------------------------------------------------------------------------- defaults never drift
def test_schema_defaults_equal_the_cli_and_dataclass_defaults(script) -> None:
    cli = vars(script.parse_args([]))
    halo = HaloVLMConfig()
    run = VLARunConfig()
    for root in run.sections():
        for section in sections(root):
            for field, dest in section.ARGS.items():
                assert type(section).model_fields[field].get_default(call_default_factory=True) == cli[dest], (field, dest)
            for field, name in section.OVERRIDES.items():
                default = type(section).model_fields[field].get_default(call_default_factory=True)
                expected = getattr(halo, name)
                assert (list(expected) if isinstance(expected, tuple) else expected) == default, (field, name)
    llm = run.model.llm
    assert llm.finetune == cli["llm_finetune"] and llm.lora_train_embeddings == (not cli["llm_freeze_embeddings"])


def test_base_yaml_is_exactly_the_cli_defaults(script) -> None:
    cli = vars(script.parse_args([]))
    from_yaml = vars(script.parse_args(["--config", str(CONFIGS / "base.yaml")]))
    assert {k: v for k, v in from_yaml.items() if k not in META} == {k: v for k, v in cli.items() if k not in META}
    config = apply_model_overrides(HaloVLMConfig(), from_yaml["model_overrides"])
    assert config == HaloVLMConfig()


def test_dual_system_section_mirrors_every_dual_system_option() -> None:
    options = {f.name for f in dataclasses.fields(DualSystemConfig)} - {"name", "notes"}
    assert set(DualSystemSection.model_fields) - {"preset"} == options


# ---------------------------------------------------------------------------- every config
@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_every_config_loads_and_builds_a_model_config(script, path: Path) -> None:
    args = script.parse_args(["--config", str(path)])
    config = HaloVLMConfig(action_dim=args.action_dim, state_dim=args.state_dim,
                           action_chunk_size=args.action_chunk_size, proj_type=args.projector_type)
    apply_model_overrides(config, args.model_overrides)
    dual = resolve_dual_system_config(args.dual_system, parse_overrides(args.dual_system_option) or None)
    if path.parent.name == "dual_system":
        assert dual is not None and args.dual_system == ("custom" if "custom" in path.stem else path.stem)
    if args.llm_backbone:
        from hale_core.nn.llm import is_registered

        assert is_registered(args.llm_backbone)


def test_yaml_values_reach_the_script_and_flags_override_them(script) -> None:
    args = script.parse_args(["--config", str(CONFIGS / "examples" / "llm_gemma4_e4b.yaml")])
    assert (args.llm_backbone, args.llm_attn_implementation, args.projector_type) == ("gemma4-e4b", "eager", "qformer")
    assert args.vision_encoder == "siglip" and args.llm_finetune == "lora"
    args = script.parse_args(["--config", str(CONFIGS / "overfit.yaml"), "--epochs", "3", "--dual_system", "lcb",
                              "--model_option", "emb_dim=96"])
    assert (args.epochs, args.dual_system, args.batch_size) == (3, "lcb", 2)
    assert args.model_overrides["emb_dim"] == 128 and parse_model_options(args.model_option) == {"emb_dim": 96}


def test_yaml_and_cli_dual_system_options_merge(script) -> None:
    args = script.parse_args(["--config", str(CONFIGS / "dual_system" / "openhelix.yaml"),
                              "--dual_system_option", "async_steps=60"])
    cfg = resolve_dual_system_config(args.dual_system, parse_overrides(args.dual_system_option))
    assert (cfg.name, cfg.prealign_steps, cfg.async_steps, cfg.system2_training) == ("openhelix", 2000, 60, "prompt_tuning")


@pytest.mark.parametrize("body, match", [
    ("model: {emb_dimm: 3}", "emb_dimm"),
    ("model: {llm: {backbone: gpt-9}}", "unknown LLM backbone"),
    ("model: {vision: {encoder: dino}}", "encoder"),
    ("model: {dual_system: {preset: custom, system2_training: prompt_tuning, latent: pooled}}", "prompt_tuning"),
    ("model: {dual_system: {preset: helix}}", "preset"),
    ("train: {steps: 10}", "steps"),
])
def test_invalid_configs_are_rejected(tmp_path: Path, body: str, match: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(f"inherits: {CONFIGS / 'base.yaml'}\n{body}\n")
    with pytest.raises(ValidationError, match=match):
        load_vla_config(path)


def test_model_options_are_typed_and_validated() -> None:
    opts = parse_model_options(["emb_dim=256", "use_moe=false", "action_hidden_dims=128,64",
                                "llm_backbone=none", "visual_loss_weight=2.5"])
    assert opts == {"emb_dim": 256, "use_moe": False, "action_hidden_dims": [128, 64], "llm_backbone": None,
                    "visual_loss_weight": 2.5}
    config = apply_model_overrides(HaloVLMConfig(), opts)
    assert config.action_hidden_dims == (128, 64) and config.use_moe is False
    with pytest.raises(ValueError):
        parse_model_options(["not_a_field=1"])
    with pytest.raises(ValueError, match="proj_type"):
        apply_model_overrides(HaloVLMConfig(), {"proj_type": "perceiver"})


# ---------------------------------------------------------------------------- end to end
def test_train_runs_a_dual_system_from_yaml(script, tmp_path: Path) -> None:
    """A YAML that inherits overfit.yaml (-> base.yaml) drives a real HaloVLM + OpenHelix run."""
    from models.halo_vla import HaloVLM

    script.HaloVLM = HaloVLM

    class Rows(list):
        tokenizer = type("Tok", (), {"pad_token_id": 0, "decode": staticmethod(lambda *a, **k: "")})()

    def build_loader(img_size: int, **_kw):
        g = torch.Generator().manual_seed(0)
        rows = Rows({"images": torch.rand(2, 3, img_size, img_size, generator=g),
                     "input_ids": torch.randint(1, 50, (8,), generator=g),
                     "attention_mask": torch.ones(8, dtype=torch.long),
                     "labels": torch.randint(1, 50, (8,), generator=g), "actions": torch.randn(4, 4, generator=g),
                     "action_mask": torch.ones(4), "states": torch.randn(1, 3, generator=g)} for _ in range(8))
        return DataLoader(rows, batch_size=2)

    script.build_eo_dataloader = build_loader
    run = tmp_path / "run.yaml"
    run.write_text(f"""inherits: {CONFIGS / 'overfit.yaml'}
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
    assert (config.emb_dim, config.img_size, config.use_moe, config.dual_system) == (32, 32, False, "openhelix")
    assert config.dual_system_options["prealign_steps"] == 2
    assert any(k.startswith("system1.") for k in ckpt["model_state_dict"])
    assert f"# command-line overrides: --config {run}" in (tmp_path / "run_config.yaml").read_text()
    saved = load_vla_config(tmp_path / "run_config.yaml")  # provenance: resolved YAML next to the checkpoints
    assert saved.variant == "tiny_openhelix" and saved.model.dual_system.preset == "openhelix"
    assert saved.model.world_model.enabled is False  # inherited from overfit.yaml
