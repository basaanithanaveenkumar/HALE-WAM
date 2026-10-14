# Vision encoders, validation and trainable-only optimisation

These bring Halo-VLA to parity with Hale-VLM. Together with Ray Train / WebDataset
(`docs/distributed.md`) and pretrained LLM backbones (`docs/llm_backbone.md`), the VLA now has
every training feature the VLM has.

| Hale-VLM | Halo-VLA |
|---|---|
| `vision.encoder: siglip \| clip \| local` (+ freeze, partial freeze) | `--vision_encoder vit \| siglip \| clip \| local` |
| `model.llm.backbone` + LoRA / full / frozen | `--llm_backbone` + `--llm_finetune` |
| `VLMEvaluator` / metrics at eval epochs | `VLAEvaluator`: `--val_fraction`, `--eval_every` |
| `VLMTrainer`: optimiser over trainable parameters only | `AdamW` over `requires_grad` parameters only |
| Ray Train, WebDataset | `--ray*`, `--webdataset*` |

## Vision encoder

```bash
python scripts/train.py --vision_encoder siglip                       # google/siglip-base-patch16-224, frozen
python scripts/train.py --vision_encoder clip --no-vision_freeze      # fine-tune CLIP ViT-B/16
python scripts/train.py --vision_encoder local --vision_backbone dino_vit_base \
    --vision_freeze_up_to_stage 8 --vision_pretrained_path ckpt/dinov2_b.pt
```

* `vit` (default): the built-in `VisTransformer`, trained from scratch, unchanged.
* `siglip` / `clip`: pretrained Hub towers (`--vision_model_id` to change the repo). The VLA
  datasets normalise with ImageNet statistics; the encoder re-normalises to its own statistics
  (read from the repo's image processor), so pretrained features are not distorted. `img_size`
  must equal the tower's native resolution (checked).
* `local`: any `hale_core.nn.vision` backbone by name (ViT / DINOv2 / EVA-02 / Swin / ConvNeXt / …),
  through `models/vision_backbone.py`, which was already in the repo but never used.
* Pretrained encoders are frozen by default (`--no-vision_freeze` trains them); a frozen encoder
  stays in eval mode under `model.train()` (no dropout, fixed BatchNorm statistics).
* The encoder's token width sets the image projector's input, and its token count (measured with
  one dummy forward) sets `num_image_tokens_per_image`. The language loss uses that count to
  offset the text positions, so CNN grids and other patch sizes stay correct.

## Validation

```bash
python scripts/train.py ... --val_fraction 0.05 --eval_every 1 --eval_max_batches 50
```

`--val_fraction` holds out a fixed random share of the training samples (same on every rank and
on resume; the WebDataset fingerprint includes it). After every `--eval_every` epochs rank 0 runs
`training.VLAEvaluator` with the unwrapped model:

* `val/loss_total`, `val/loss_lang`, `val/loss_act`, `val/loss_vis`;
* actions sampled with the flow decoder (Euler, `--eval_flow_steps`) against the ground truth on
  unpadded steps: `val/action_l1` (+ `_t1.._t3`), `val/action_mse`, `val/action_rmse`,
  `val/chunk_acc_{0.05,0.10,0.20}`, `val/pred_smoothness` vs `val/target_smoothness`.

The metrics are logged, written to TensorBoard and included in Ray Train reports.

Two fixes came with this:

* `src/Halo_VLA/metrics` used absolute `Halo_VLA.metrics.*` imports, which fail on the import path
  the training script sets up; it now uses relative imports.
* `TrajectorySmoothnessMetric` sliced `a[t]` as `a[:, :-1]` instead of `a[:, :-2]`, so it crashed
  on any trajectory of 3+ steps. It now computes true second differences (regression test: a
  quadratic trajectory gives exactly 4).

## Trainable-only optimisation

`AdamW` receives only parameters with `requires_grad` (and gradient clipping only touches those),
so frozen encoders and frozen / LoRA-adapted LLM weights get no optimiser state. Resuming a
checkpoint written before this change keeps its weights and restarts the optimiser moments, with
a warning.

Tests: `tests/test_vision_encoders_and_eval.py`,
`tests/test_training_runtime.py::test_validation_split_runs_the_evaluator_each_epoch`.
