<div align="center">

# Halo-WAM

### A compact Vision-Language-Action model with an integrated world model

*Perceive · Reason · Act · Imagine*

[![GitHub](https://img.shields.io/badge/GitHub-HALE--WAM-181717?logo=github&logoColor=white)](https://github.com/basaanithanaveenkumar/HALE-WAM)
[![Project Page](https://img.shields.io/badge/🌐_Project-Page-4A90D9)](https://basaanithanaveenkumar.github.io/HALE-WAM/)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b?logo=arxiv&logoColor=white)](https://github.com/basaanithanaveenkumar/HALE-WAM/blob/main/paper/main.tex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

Halo-VLA is a single unified model that handles **language grounding**, **robot action prediction**, and **future-frame imagination** in one forward pass. It takes camera frames and proprioceptive state as input, generates a natural-language description of the scene, predicts a continuous action trajectory, and synthesises what the world will look like next — all jointly trained end-to-end.

<div align="center">

### Checkpoint — epoch 24 000 · step 24 000

| Training visualisation | Diffusion (noise → frame) |
|:---:|:---:|
| ![Training visualisation](assets/training_viz_sample1.gif) | ![DiT diffusion](assets/diffusion_viz_sample1.gif) |
| Context scroll → text reveal → action trajectory → predicted future frames | DiT velocity-field denoising: pure noise → predicted RGB frame |

</div>

---

## Architecture

### Key components

| Module | What it does |
|---|---|
| **ViT** | 4-layer Vision Transformer encodes each frame into patch embeddings |
| **Decoder Transformer** | 8-layer causal transformer with DeepSeekMoE FFN; fuses vision, state and text |
| **Flow Matching Action Decoder** | Predicts robot trajectories via ODE integration; learns the velocity field `v*(x,t) = x₁ − x₀` |
| **DiT World Model** | Diffusion Transformer predicts future RGB frames conditioned on world-action tokens and text embeddings |
| **ALBERT-style factored embeddings** | Vocabulary embedding bottlenecked through a 128-dim projection; cuts embedding memory ~10× vs full tables |

### Flow matching — how action prediction works

```
Training                              Inference
──────────────────────────────────    ──────────────────────────
x₀ ~ N(0, I)  (noise)                x  = x₀  (start from noise)
x₁ = target action chunk             dt = 1 / num_steps
t  ~ U(0, 1)                         for i in range(num_steps):
x_t = (1-t)·x₀ + t·x₁                   v = DiT(x, t, cond)
v*  = x₁ − x₀                            x = x + v · dt
loss = MSE(DiT(x_t, t, cond), v*)    return x  # ≈ clean action chunk
```

The same formulation drives the **DiT world model** — it learns to denoise random noise into a predicted future video frame, conditioned on the decoded world-action tokens.

### DiT world model — per-frame conditioning pipeline

Each predicted future frame gets a **unique conditioning vector** built from four stacked signals:

```
world_action_tokens[frame_offset + i]
        │
        ▼
  Cross-Attention ←── text embeddings (keys / values)
        │
        + sinusoidal frame-index embedding  (frame i vs frame j always differ)
        │
        + context_frame_enc(last_seen_frame)  (SVD-style pixel-level anchor)
        │
        ▼
  CFG dropout (p=0.1 during training)  →  cfg_null_context
        │
        ▼
  DiT adaLN: c = sigmoid(W·ctx) · t_emb + ctx_emb
```

At inference the DiT integrates via **Heun's 2nd-order ODE** (halves truncation error vs Euler) with optional **Classifier-Free Guidance** for sharper predictions.

Training uses three auxiliary losses on top of the flow-matching MSE to combat blurring:
- **VGG perceptual loss** — feature-space MSE at relu1_2 and relu2_2
- **SSIM loss** — structural similarity on the estimated clean frame
- **Temporal smoothness** — L2 between consecutive predicted velocity fields

---

## Project structure

```
Han-WAM/
├── src/Halo_VLA/models/
│   ├── halo_vla.py              # Main model: forward, action loss, visual loss
│   ├── vit.py                   # Vision Transformer encoder
│   ├── image_proj.py            # Vision -> decoder projector (MLP / Q-Former / gated cross-attention)
│   ├── qformer.py               # Q-Former (BLIP-2 style learned queries)
│   ├── gated_cross_attention.py # Gated cross-attention (Flamingo style)
│   ├── transformer.py           # Decoder transformer blocks
│   ├── moe.py                   # DeepSeek Mixture-of-Experts FFN
│   ├── flow_action_decoder.py   # Flow matching action decoder
│   ├── dit_frame_prediction.py  # DiT world model (future frame prediction)
│   ├── DiT.py                   # Core Diffusion Transformer with adaLN
│   ├── lm_head.py               # Language model head
│   └── state_encoder.py         # Proprioceptive state encoder
├── config/
│   ├── model_config.py          # HaloVLMConfig dataclass (all hyperparameters)
│   ├── tokens.py                # Special token definitions
│   └── special_tokens.json      # Token IDs for <image>, <action>, <state>, <world_video>
├── dataloader/
│   ├── airoa_moma_dataset.py    # AiroaMoMa video dataset (lerobot parquet format)
│   └── eo_dataset.py            # EO dataset loader
├── scripts/
│   ├── train.py                 # Training loop with visualisation
│   └── visualize.py             # Standalone inference visualiser
└── assets/
    ├── training_viz.gif         # Training visualisation example
    └── diffusion_viz.gif        # DiT denoising example
```

---

## Installation

```bash
git clone https://github.com/basaanithanaveenkumar/Han-WAM.git
cd Han-WAM
uv pip install -e .          # recommended
# or: pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, CUDA GPU recommended.

---

## Training

### Overfit on a single clip (sanity check)

```bash
python scripts/train.py \
  --dataset moma \
  --moma_data_root /path/to/moma_dataset \
  --max_samples 1 \
  --epochs 5000 \
  --batch_size 1 \
  --lr 1e-4 \
  --vis_every 100
```

### Full training run

```bash
python scripts/train.py \
  --dataset moma \
  --moma_data_root /path/to/moma_dataset \
  --batch_size 4 \
  --epochs 100 \
  --lr 3e-4 \
  --moma_num_frames 5 \
  --num_predict_frames 5 \
  --ckpt_dir checkpoints/run1 \
  --tensorboard_dir runs/run1
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--moma_num_frames` | 5 | Context frames fed to the model |
| `--num_predict_frames` | 5 | Future frames the DiT world model predicts |
| `--action_chunk_size` | 16 | Timesteps per predicted action chunk |
| `--action_dim` | 32 | Proprioceptive action dimensionality |
| `--moma_frame_stride` | 25 | Stride between sampled video frames |
| `--visual_loss_weight` | — | Weight on DiT frame-prediction loss |
| `--perceptual_weight` | 0.1 | VGG perceptual loss weight (0 = off) |
| `--ssim_weight` | 0.4 | SSIM loss weight (0 = off) |
| `--temporal_weight` | 0.2 | Temporal smoothness loss weight (0 = off) |
| `--grad_accum_steps` | 1 | Accumulate gradients over N micro-batches; effective batch = `batch_size × grad_accum_steps` |
| `--resume` | — | Path to checkpoint to resume from |
| `--vis_every` | — | Save visualisation GIFs every N optimizer steps |

### Memory-constrained training (gradient accumulation)

Simulate a large effective batch on a single GPU by accumulating gradients across micro-batches. Scale the learning rate proportionally.

```bash
# 8 GB GPU: effective batch 16 at batch_size 1
python scripts/train.py \
  --dataset moma --moma_data_root /path/to/moma \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --lr 4e-4 \
  --epochs 20
```

### Resuming from a checkpoint

```bash
python scripts/train.py \
  --dataset moma --moma_data_root /path/to/moma \
  --resume checkpoints/run1/halo_vla_epoch10.pt \
  --epochs 50
```

Restores model weights, optimizer, scheduler, and AMP scaler. Training continues from the next epoch stored in the checkpoint.

### Visualisation at inference

```bash
python scripts/visualize.py \
  --checkpoint checkpoints/run1/model_best.pt \
  --moma_data_root /path/to/moma_dataset \
  --output_dir vis_output \
  --diffusion_steps 50
```

---

## Model configuration

All hyperparameters live in a single dataclass — `HaloVLMConfig` in `config/model_config.py`.

```python
from config import HaloVLMConfig

config = HaloVLMConfig(
    emb_dim=512,
    dec_num_layers=8,
    dec_num_heads=16,
    use_moe=True,
    moe_num_routed_experts=4,
    moe_top_k=2,
    action_chunk_size=16,
    flow_num_ode_steps=24,
    dit_hidden_size=512,
    dit_depth=8,
    dit_num_sample_steps=50,
    num_visual_predict_frames=5,
)
```

### Image projector: MLP, Q-Former or gated cross-attention

`proj_type` chooses how ViT patch features enter the decoder:

| `proj_type` | Decoder tokens per image | Config fields |
|---|---|---|
| `"mlp"` (default) | one per ViT patch | `proj_vision_dim`, `proj_llm_dim` |
| `"qformer"` | `qformer_num_queries` | `qformer_hidden_dim`, `_num_layers`, `_num_heads`, `_cross_attention_freq`, `_ffn_mult`, `_dropout` |
| `"gated_cross_attention"` | `gated_xattn_num_latents` | `gated_xattn_hidden_dim`, `_num_layers`, `_num_heads`, `_ffn_mult`, `_dropout`, `_gate_init` |

```python
config = HaloVLMConfig(proj_type="qformer", qformer_num_queries=32, qformer_num_layers=2)
config = HaloVLMConfig(proj_type="gated_cross_attention", gated_xattn_num_latents=32, gated_xattn_gate_init=0.0)
```

* **Q-Former** — learned queries self-attend and cross-attend to the patch features (every `qformer_cross_attention_freq`-th layer).
* **Gated cross-attention** — learned latents read the patches through `tanh(alpha) * CrossAttn(...)` and `tanh(alpha) * FFN(...)`
  residuals with learned gates. `gated_xattn_gate_init=0.0` is the Flamingo identity start (image enters as the gates open);
  the default `1.0` lets image information flow from step 0.
* `config.num_image_tokens_per_image` gives the decoder tokens one image occupies; `scripts/train.py` and `scripts/inference.py`
  use it to align the language loss. Train with `--projector_type qformer --num_query_tokens 32`.
* Checkpoints trained with one `proj_type` cannot be loaded into another. The default `"mlp"` keeps the old state-dict keys.

---

## What the visualisations show

**Training GIF** — saved periodically during training, each GIF walks through three phases:

1. **Context scroll** — the N input camera frames the model actually sees
2. **Text reveal** — the model's generated description, token by token
3. **Action trajectory** — GT (solid) vs predicted (dashed) action dimensions, animated step by step
4. **Future frames** — GT future frame alongside the DiT's predicted future frame

**Diffusion GIF** — shows the DiT world model's denoising process for a single future frame: starting from pure Gaussian noise at `t=0` and integrating forward step by step to `t=1`, where the final frame is the model's prediction of the next scene.

---

## Documentation

Detailed developer documentation lives in [`docs/`](docs/):

| Document | Contents |
|---|---|
| [`docs/getting-started.md`](docs/getting-started.md) | Install, smoke test, first training runs |
| [`docs/architecture.md`](docs/architecture.md) | Mermaid architecture diagrams (system, flow head, DiT conditioning, sampler, training step) |
| [`docs/configuration.md`](docs/configuration.md) | Every `HaloVLMConfig` field and training flag |
| [`docs/world_model.md`](docs/world_model.md) | Full technical write-up of the DiT world model — per-frame conditioning, CFG, Heun solver, auxiliary losses, and bug fixes |
| [`docs/blog/`](docs/blog/README.md) | Blog posts, including *Teaching a robot policy to imagine* |

### Paper, project page and skills

| | |
|---|---|
| arXiv paper | [`paper/main.tex`](paper/main.tex) — build with `make -C paper` |
| Project page | [basaanithanaveenkumar.github.io/HALE-WAM](https://basaanithanaveenkumar.github.io/HALE-WAM/) ([source](project-page/index.html), deployed by `.github/workflows/pages.yml`) |
| Claude Code skills | [`.claude/skills/`](.claude/skills) — `hale-wam-dev`, `hale-wam-train`, `hale-wam-world-model`, `hale-publish` |

---

## License

MIT — see [LICENSE](LICENSE).
