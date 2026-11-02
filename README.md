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

VLMBox targets Python 3.11+ and Transformers 5.x. Install a PyTorch build compatible with your platform and accelerator, then install VLMBox:

```bash
uv sync --extra dev
```

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

The runtime dependencies include PyTorch, torchvision, and Pillow for model and image processing. Models are downloaded through ModelScope's default cache. `VLM.from_pretrained()` accepts a ModelScope model ID only. The runtime uses `AutoProcessor` and `AutoModelForMultimodalLM`, with `dtype="auto"` and `device_map="auto"`. Model repository code is never executed: `trust_remote_code` is fixed to `False`.

The generic `TransformersAdapter` has been validated against Qwen2.5-VL-3B, InternVL3.5-2B-HF, and SmolVLM-500M. Full CPU load and text-generation smoke tests passed for all three; image-input smoke tests passed for SmolVLM and the InternVL processor.

## Development

```bash
uv run pytest
uv run ruff check .
```
