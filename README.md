# VLMBox

A lightweight research toolbox for inspecting, steering, and optimizing vision-language models (VLMs).

VLMBox builds on PyTorch and Transformers. It provides a small, model-oriented API for common research operations without introducing a separate model runtime or preprocessing stack.

## Features

- Load multimodal models from ModelScope through Transformers.
- Prepare single or batched conversations and generate outputs.
- Capture and intervene on decoder activations.
- Compute representation differences, projections, and similarities.
- Apply and optimize universal image perturbations and patches.
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

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image briefly."},
    ],
}]

inputs = vlm.prepare(messages)
sequences = vlm.generate(inputs, max_new_tokens=64, do_sample=False)
new_tokens = sequences[:, inputs["input_ids"].shape[1]:]
print(vlm.processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

`VLM.from_pretrained()` accepts a ModelScope model ID. Use `local_files_only=True` to require an existing local cache and disable downloads.

## Documentation

- [Architecture and design](docs/architecture.md)
- [Public API reference](docs/api.md)
- [Image preprocessing and gradients](docs/preprocessing.md)
- [Visual Input Rendering Protocol (VIRP)](docs/protocols/visual-input-rendering.md)
- [ImageRenderer YAML example](examples/image_render.yaml)

Runnable examples are in [`examples/`](examples/).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

See [RELEASING.md](RELEASING.md) for package release procedures.

## License

MIT. See [LICENSE](LICENSE).
