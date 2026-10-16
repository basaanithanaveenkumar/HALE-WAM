# ext_pkgs — vendor deployment SDKs (third-party, read-only)

Shallow git submodules (`shallow = true` in `.gitmodules`, Git LFS objects not fetched) of the
upstream toolchains the deployment pipelines (`docs/deploy.md`) target. They are **reference
sources** (samples, op-support tables, model recipes, C++ runtimes for on-robot apps); the Python
packages themselves are installed from their wheels via the `deploy-*` extras.

| directory | upstream | used for |
|-----------|----------|----------|
| `TensorRT` | NVIDIA/TensorRT | ONNX parser, plugins, `trtexec`, samples (TensorRT 10/11) |
| `TensorRT-LLM` | NVIDIA/TensorRT-LLM | System-2 LLM serving (PyTorch backend, `trtllm-serve`) |
| `TensorRT-Edge-LLM` | NVIDIA/TensorRT-Edge-LLM | on-device LLM/VLM runtime for Jetson Thor / DRIVE |
| `Model-Optimizer` | NVIDIA/Model-Optimizer | AutoCast, ONNX PTQ (INT8/FP8/INT4), LLM PTQ |
| `Torch-TensorRT` | pytorch/TensorRT | dynamo partial lowering |
| `mlx`, `mlx-lm`, `mlx-vlm` | ml-explore/mlx, ml-explore/mlx-lm, Blaizzy/mlx-vlm | Apple Silicon runtime, `.mlxfn` C++/Swift loaders, LLM/VLM converters |
| `AMDMIGraphX` | ROCm/AMDMIGraphX | AMD GPU graph compiler (`migraphx-driver`) |
| `Quark` | amd/Quark | Ryzen AI NPU quantization (A16W8, BF16) |
| `RyzenAI-SW` | amd/RyzenAI-SW | Vitis AI EP examples / NPU deployment |
| `ai-hub-models` | qualcomm/ai-hub-models | AI Hub recipes, Snapdragon / QCS app templates |
| `executorch` | pytorch/executorch | QNN backend, Android runtime (`.pte`) |

```bash
git submodule update --init --depth 1 ext_pkgs/<name>   # fetch one (never --recursive: executorch nests many)
git submodule update --remote --depth 1 ext_pkgs/<name> # move to the upstream head
```
