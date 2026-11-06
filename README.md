# VLMBox

A lightweight library for controlled experiments on vision-language models (VLMs): inspect decoder representations, intervene on activations, and optimize reusable image interventions.

VLMBox provides `VLM` for unified model operations and `Renderer` for reproducible visual inputs. Optional image optimization lives in `vlmbox.trainer`; numerical helpers live in `vlmbox.utils`. PyTorch and Transformers own execution, preprocessing, and generation. Experiment code owns datasets, losses, evaluation, and reporting.

The core components are `VLM` (`model/`), `Renderer` (`renderer/`), and `Trainer` (`trainer.py`). Model loading, layout adaptation, activation operations, and generation traces live together in `model/`. See the [architecture](docs/architecture.md) for package boundaries.

`Renderer` is a PyTorch module: `renderer(host, instruction)` returns a differentiable Tensor, while `renderer.render(host, instruction)` returns a PIL image. Training uses standard `.parameters()`, `.to()`, and a bounded `.project_()` update.

## Features

- Load multimodal models from ModelScope through Transformers.
- Prepare single or batched conversations and generate outputs.
- Capture and intervene on decoder activations.
- Compute representation differences, projections, and similarities.
- Apply and optimize universal image perturbations and patches.
- Save and reload learned image interventions independently of host images.
- Record structured autoregressive generation traces.
- Configure random seeds and deterministic PyTorch execution.

## Installation

```bash
pip install vlmbox
```

For the runnable image-training example, install its optional dependency:

```bash
pip install 'vlmbox[examples]'
```

VLMBox requires Python 3.11 or newer. Install a PyTorch build compatible with your CUDA runtime and hardware before running GPU workloads. Model checkpoints are resolved through ModelScope.

## Quick start

```python
from PIL import Image
from vlmbox import VLM

vlm = VLM.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")

with Image.open("image.jpg") as source:
    image = source.convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]

inputs = vlm.prepare(messages)
sequences = vlm.generate(inputs, max_new_tokens=64, do_sample=False)
new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
print(vlm.processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

All layer indices use zero-based decoder blocks; capture and traces return block outputs before final model normalization.

For advanced loading needs, `VLM(model, processor)` also wraps already-loaded native objects, with optional `adapter=` for custom decoder layouts. `VLM.from_pretrained()` accepts a ModelScope model ID. Use `local_files_only=True` to require an existing local cache and disable downloads.

## Documentation

- [Architecture and design](docs/architecture.md)
- [Public API reference](docs/api.md)
- [Research adapters and API migration](docs/research-adapters.md)
- [Image preprocessing and gradients](docs/preprocessing.md)
- [Visual Input Rendering Protocol (VIRP)](docs/protocols/visual-input-rendering.md)
- [Renderer YAML example](examples/image_render.yaml)

Runnable examples are in [`examples/`](examples/).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

The default test suite includes CPU regressions for the training example's public API. To validate native preprocessing, generation traces, and image training on the three supported reference models, use a CUDA machine with their ModelScope checkpoints cached locally:

```bash
VLMBOX_RUN_GPU_INTEGRATION=1 uv run pytest tests/integration
```

See [RELEASING.md](RELEASING.md) for package release procedures.

## License

MIT. See [LICENSE](LICENSE).
