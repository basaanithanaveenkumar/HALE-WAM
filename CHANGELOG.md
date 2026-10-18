# Changelog

## Unreleased

- Added Q-Former and gated cross-attention image projectors, selected with `HaloVLMConfig.proj_type` (`mlp` | `qformer` | `gated_cross_attention`), plus `--projector_type` / `--num_query_tokens` in `scripts/train.py`
- Production-spec pass 2: registries replace the if/elif chains on dual-system presets (`DUAL_SYSTEM_PRESETS`), latent extractors (`LATENT_EXTRACTORS`), System 1 policies (`SYSTEM1_POLICIES`, a strategy per policy kind shared by training, sampling and the deploy graphs), bridge projectors, image projectors (`IMAGE_PROJECTORS`), vision encoders (`VISION_ENCODERS`) and training datasets (`TRAIN_DATASETS`); `scripts/train.py` delegates to `CheckpointManager`, `build_model_config` and `build_train_loader`; `dual_system.config` keeps `PRESETS` / `resolve_dual_system_config` / `parse_overrides` importable (they live in `dual_system.presets` / `dual_system.resolution`)
- Tests split into `tests/unit`, `tests/integration` and `tests/support`; the run-config test no longer globs `configs/deploy`, and `configs/deploy/_shared/parts` is tracked (the build-directory `.gitignore` rule had hidden it)
- Google-style docstrings and type hints on the public API, `docs/registries.md` and mermaid diagrams, ruff configuration and formatting
