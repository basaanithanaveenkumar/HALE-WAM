# Roadmap configs (not loadable yet)

These YAML files describe runs whose code paths do not exist in Halo-VLA yet. They are kept as
specifications, outside `configs/`, because `scripts/train.py --config` validates every key
against `config/run_config.py::VLARunConfig` and would reject them. `base.yaml` here is the
earlier flat layout that they inherit from.

| file | needs |
|---|---|
| `pretrain_vla.yaml`, `mid_train_vla.yaml`, `post_train_vla.yaml` | a training adapter for `dataloader/stages/` (its registries return raw Hub rows, not model batches) and a `data.dataset: stage_*` source |
| `vlm_pretrain.yaml`, `vlm_mid_train.yaml`, `vlm_post_train.yaml` | an image-text-only training mode (no action / video losses) |
| `eval_libero.yaml`, `eval_metaworld.yaml` | simulator rollouts (LIBERO, Meta-World); validation today is held-out losses + action metrics (`eval.val_fraction`) |
| `deploy_gpu.yaml`, `deploy_local.yaml`, `deploy_onnx.yaml`, `deploy_robot.yaml` | a deployment runner (`torch.compile` / ONNX export, ROS 2 control loop) |
| `gwm_vla.yaml` | the multi-view geometric world-model architecture |

When one of these lands, rewrite its file in the `configs/base.yaml` schema and move it back.
