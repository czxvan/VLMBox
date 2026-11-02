# VLMBox

A lightweight, modular Python research toolbox for vision-language models.

VLMBox focuses on model internals and research operations rather than serving or training infrastructure.

## Design

- **Model** — load ModelScope checkpoints through Transformers.
- **Adapter** — resolve language decoder components without hard-coding a model family into research code.
- **Activation** — capture and intervene on internal activations (under development).
- **Representation** — direction extraction and representation geometry (under development).
- **Image** — image perturbation and patch operations (under development).
- **Generation** — generation results and internal traces (under development).
- **Repro** — deterministic execution and run metadata (under development).

## Installation

VLMBox targets Python 3.11+ and uses one CUDA 13.0 PyTorch build for NVIDIA RTX 3090 (Ampere) and RTX 4090 (Ada). The supported Linux research machines must have an NVIDIA driver >=580. Use a machine-local environment rather than placing `.venv` on the shared workspace:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/VLMBox"
uv sync --locked --extra dev
```

The Tsinghua PyPI mirror and PyTorch CUDA 13.0 index are both declared in `pyproject.toml`, so dependency resolution does not depend on machine-specific uv configuration. `uv.lock` pins the resolved wheels. Use the same commands on each machine. Do not manually install a CPU-only PyTorch build or use `pip`/Conda to mutate this environment.

## Quick start

```python
from PIL import Image
from vlmbox import VLM

vlm = VLM.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
image = Image.open("example.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
inputs = vlm.prepare(messages)
output_ids = vlm.generate(inputs, max_new_tokens=128)
new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
response = vlm.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
print(response)
```

Models are downloaded through ModelScope's default cache. `VLM.from_pretrained()` accepts a ModelScope model ID only. The runtime uses `AutoProcessor` and `AutoModelForMultimodalLM`, with `dtype="auto"`, `device_map="auto"`, and fixed `trust_remote_code=True`. Only load model repositories whose code you trust.

The generic `TransformersAdapter` has been validated against Qwen2.5-VL-3B, InternVL3.5-2B-HF, and SmolVLM-500M. GPU smoke tests are the required runtime validation path; CPU-only tests are limited to isolated unit tests.

## Development

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/VLMBox"
uv run pytest
uv run ruff check .
```
