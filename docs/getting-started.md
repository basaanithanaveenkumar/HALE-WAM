# Getting started

## Install

```bash
git clone https://github.com/basaanithanaveenkumar/HALE-WAM && cd HALE-WAM
uv venv && source .venv/bin/activate
uv pip install -e .
export PYTHONPATH=.:src/Halo_VLA
```

## Smoke test (CPU)

```python
import torch
from config import HaloVLMConfig
from models.halo_vla import HaloVLM

c = HaloVLMConfig()
m = HaloVLM(c).eval()
ids = torch.tensor([[c.image_token_id, c.image_token_id, 1, 2, c.state_token_id, 3,
                     c.action_token_id, c.world_video_token_id]])
with torch.no_grad():
    logits, action_h, vis_ctx, wv_h = m(torch.randn(1, 2, 3, 224, 224), ids,
                                        torch.ones_like(ids), torch.randn(1, 1, 32))
    actions = m.sample_actions(action_h)                      # [1, 1, 16, 7]
    frames = m.predict_visual_future(torch.rand(1, 2, 3, 224, 224), vis_ctx,
                                     num_frames=2, world_video_query_hiddens=wv_h,
                                     num_ode_steps=4)["rgb"]  # [1, 2, 3, 224, 224]
print(logits.shape, actions.shape, frames.shape)
```

The default model has 176.3M parameters.

## Training

```bash
# EO-Data1.5M from the Hugging Face Hub
python scripts/train.py --dataset eo --subset interleave-temporal --max_samples 200

# AIRoA MoMa (local clone with episodes.jsonl + videos/)
python scripts/train.py --dataset moma --moma_data_root /path/to/moma \
  --moma_num_frames 5 --num_predict_frames 5 --batch_size 1 --grad_accum_steps 8
```

Checkpoints go to `--ckpt_dir`. TensorBoard logs go to `--tensorboard_dir`, and GIFs are
written every `--vis_every` steps.

## Visualise

```bash
python scripts/visualize.py --checkpoint checkpoints/model_best.pt \
  --moma_data_root /path/to/moma --output_dir vis_output --diffusion_steps 50
```
