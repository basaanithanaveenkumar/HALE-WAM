# HALE-WAM documentation

HALE-WAM (Halo-VLA world-action model) is one transformer that **perceives** camera frames and
robot state, **reasons** in language, **acts** with a flow-matching action head and
**imagines** future frames with a DiT world model.

| Page | Contents |
|---|---|
| [Getting started](getting-started.md) | install, smoke test, first runs |
| [Architecture](architecture.md) | Mermaid diagrams: system, token routing, flow head, DiT conditioning, training step |
| [Configuration](configuration.md) | `HaloVLMConfig` fields and training flags |
| [World model reference](world_model.md) | detailed write-up of the DiT head, losses and bug fixes |
| [Blog](blog/README.md) | long-form posts |

Also: the [paper](../paper/main.tex) and the [project page](../project-page/index.html).
