# Changelog

## Unreleased

- Added Q-Former and gated cross-attention image projectors, selected with `HaloVLMConfig.proj_type` (`mlp` | `qformer` | `gated_cross_attention`), plus `--projector_type` / `--num_query_tokens` in `scripts/train.py`
