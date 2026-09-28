---
name: hale-wam-dev
description: Set up, navigate and smoke-test HALE-WAM (the Halo-VLA world-action model — VLA + flow-matching actions + DiT future-frame prediction). Use when starting work in this repo, when imports fail, when locating a component, or before editing src/Halo_VLA/models.
---

# HALE-WAM development

HALE-WAM extends the Hale-VLA design with a **flow-matching action decoder** and a
**DiT world model** that predicts future RGB frames. The model class is `HaloVLM`
(`src/Halo_VLA/models/halo_vla.py`), configured by `HaloVLMConfig`
(`config/model_config.py`).

## Environment

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
export PYTHONPATH=.:src/Halo_VLA     # modules import `models.*` and `config`
```

## Code map

| Path | What it is |
|---|---|
| `config/model_config.py` | `HaloVLMConfig`: dims, MoE, flow head, DiT, loss weights |
| `config/special_tokens.json` | cosmo2 tokenizer (49,152) + `<image>` 49152, `<halo_action>` 49153, `<state>` 49154, `<halo_world_video>` 49155 |
| `src/Halo_VLA/models/halo_vla.py` | `HaloVLM.forward` → `(logits, action_hiddens, visual_context_emb, world_video_query_hiddens)`; `compute_flow_loss`, `sample_actions`, `compute_visual_prediction_loss`, `predict_visual_future` |
| `src/Halo_VLA/models/flow_action_decoder.py` | `FlowActionDecoder` velocity MLP + `SinusoidalTimeEmbedding` |
| `src/Halo_VLA/models/dit_frame_prediction.py` | `RGBFramePredictor` (alias `VisualDiTPredictor`): world-action tokens, cross-attention, CFG, Heun sampler, CFM + VGG + SSIM + temporal losses |
| `src/Halo_VLA/models/DiT.py` | adaLN-Zero `DiT` with gated context fusion |
| `src/Halo_VLA/models/transformer.py`, `moe.py`, `vit.py` | decoder (MoE, padding mask, grad checkpointing), DeepSeekMoE, dense ViT |
| `src/Halo_VLA/models/video_vae.py` | `VideoVAE` (not yet wired into training) |
| `dataloader/eo_dataset.py`, `airoa_moma_dataset.py`, `droid_dataset.py` | EO-Data1.5M, AIRoA MoMa (local clone), DROID (LeRobot format) |
| `scripts/train.py` | joint training: CE + flow + visual losses, AMP, grad accumulation, resume, TensorBoard, GIFs |
| `scripts/visualize*.py`, `inference.py` | GIF/video rendering and generation |
| `docs/world_model.md` | deep technical reference for the DiT world model |

## Smoke test (CPU, about 30 s)

```bash
PYTHONPATH=.:src/Halo_VLA python - <<'PY'
import torch
from config import HaloVLMConfig
from models.halo_vla import HaloVLM
c = HaloVLMConfig(); m = HaloVLM(c).eval()
ids = torch.tensor([[c.image_token_id, c.image_token_id, 1, 2, c.state_token_id, 3,
                     c.action_token_id, c.world_video_token_id]])
with torch.no_grad():
    logits, act_h, vis_ctx, wv_h = m(torch.randn(1, 2, 3, 224, 224), ids, torch.ones_like(ids), torch.randn(1, 1, 32))
    print(logits.shape, act_h.shape, vis_ctx.shape, wv_h.shape)  # [1,400,49156] [1,1,512] [1,512] [1,1,512]
    print(m.sample_actions(act_h).shape)                          # [1,1,16,7]
PY
```

The default config has 176.3M parameters (decoder 68.8M, DiT world model 42.5M, ViT 10.0M,
token embedding and LM head 25.2M each, flow head 3.0M).

## Gotchas

- `forward` does **not** return actions. Train with `compute_flow_loss(action_hiddens, ...)`
  and sample with `sample_actions(action_hiddens, num_steps)` (Euler, 24 steps by default).
- `emb_factor_dim` (ALBERT-style factored embeddings) exists in the config but is **not
  implemented**: `token_emb` and `lm_head` are full `vocab × emb_dim` matrices.
- The ViT here is **dense** (`use_moe=False`); only the decoder uses MoE (6 routed + 2 shared, top-2).
- The DiT runs in **pixel space** at 224×224 with patch 8 (784 tokens), and frames are in `[0, 1]`.
- `visual_context_emb` averages decoder hidden states at image-patch positions over all frames
  **before** the last valid one (`pool_past_frame_embeddings_for_visual_dit`).
- `--dataset` in `train.py` accepts `eo` and `moma`; the DROID loader exists but isn't wired to the CLI.
- `tests/test_models.py` is empty; `tests/test_visual_prediction_multi_step.py` covers the DiT head.

## Before committing

```bash
pytest tests && black --check . && ruff check .
```
