# VLMBox

A small library for VLM research: prepare model inputs, inspect and edit decoder activations, trace generation, and optimize visual interventions.

| Object | Purpose |
|---|---|
| `VLM` | Model loading, preprocessing, execution, and decoder operations |
| `Renderer` | Image composition and trainable visual parameters |
| `Trainer` | Frozen-model optimization with your loss function |

## Install

In your own uv project (Python 3.11+):

```bash
uv add vlmbox
```

For local development from this checkout:

```bash
uv venv --python 3.11
uv pip install -e .
```

Local installation is editable. Native image-processing dependencies are included; checkpoints are downloaded from ModelScope.

## Quick start

```python
from PIL import Image
from vlmbox import VLM

vlm = VLM.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
image = Image.new("RGB", (224, 224), "red")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the image briefly."},
        ],
    }
]
inputs = vlm.prepare(messages)
sequences = vlm.generate(inputs, max_new_tokens=16, do_sample=False)
new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
print(vlm.processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

Use `Image.open("image.jpg").convert("RGB")` for your own image. Set `local_files_only=True` when loading to require a cached checkpoint.

## Usage and reference

- [Usage](https://github.com/czxvan/VLMBox/blob/main/docs/usage.md): batching, activations, traces, rendering, loss functions, training, and runnable experiments.
- [API](https://github.com/czxvan/VLMBox/blob/main/docs/api.md): imports, signatures, and behavior contracts.
- [Architecture](https://github.com/czxvan/VLMBox/blob/main/docs/architecture.md): the three core objects and package boundaries.
- [Rendering protocol](https://github.com/czxvan/VLMBox/blob/main/docs/protocols/visual-input-rendering.md) and [YAML configuration](https://github.com/czxvan/VLMBox/blob/main/examples/image_render.yaml).

Layer indices are **zero-based decoder blocks**. Captures and traces return direct block outputs before final model normalization.

## Development

```bash
uv sync --frozen --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

CUDA integration tests require locally cached Qwen2.5-VL, InternVL, and SmolVLM checkpoints:

```bash
VLMBOX_RUN_GPU_INTEGRATION=1 uv run pytest tests/integration
```

[Release procedure](https://github.com/czxvan/VLMBox/blob/main/RELEASING.md) · [MIT license](https://github.com/czxvan/VLMBox/blob/main/LICENSE)
