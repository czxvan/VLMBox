# VLMBox

A lightweight, modular Python research toolbox for vision-language models.

VLMBox focuses on model internals and research operations rather than serving or training infrastructure.

## Design

- **Model** — load ModelScope checkpoints through Transformers.
- **Adapter** — resolve language decoder components without hard-coding a model family into research code.
- **Activation** — context-managed capture and intervention on decoder blocks, attention, and MLP outputs.
- **Representation** — training-free direction extraction and representation geometry.
- **Image** — architecture-independent RGB conversion, resizing, cropping, and patch placement.
- **Generation** — structured generation results and opt-in internal traces.
- **Repro** — random seed control and optional deterministic execution.

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

## Activation

Capture decoder outputs without changing the model:

```python
with vlm.capture("block", layers=[8, 16, 24], positions="last") as cache:
    output_ids = vlm.generate(inputs, max_new_tokens=32)

last_layer_state = cache[24]  # [batch, 1, hidden_size]
```

Apply a temporary intervention to one decoder layer:

```python
with vlm.intervene(
    "block",
    layer=16,
    operation="add",
    value=direction,  # a torch.Tensor on the activation hidden dimension
    positions="last",
    scope="every_forward",
):
    output_ids = vlm.generate(inputs, max_new_tokens=32)
```

Supported components are `block`, `attention`, and `mlp`. Position selectors are `last`, `all`, or explicit token indices. `capture` records the first invocation of each selected module and stores the tensor at `cache[layer]`.

Intervention operations:
- `add`: `h + scale * value`.
- `replace`: substitute the selected activation with `value`.
- `project_out`: `h - scale * dot(h, value) / dot(value, value) * value`.
- `zero`: set the selected activation to zero.

The value tensor must broadcast to the selected activation shape. `scale` is supported only by `add` and `project_out`. `first_forward` applies once per target layer (normally prompt prefill with cached generation); `every_forward` reapplies on every module invocation, including decoding. Nested hooks execute in registration order: enter intervention before capture to capture the edited activation, or capture before intervention to record the original activation. Hooks are removed when the context exits, including when an exception occurs. Captured tensors retain their autograd graph; detach them explicitly when graph retention is not needed.

## Representation

Representation operations are pure PyTorch functions and do not train a classifier:

```python
from vlmbox import (
    angle,
    cosine_similarity,
    mean_difference,
    normalize_direction,
    project,
    projection_coefficient,
    projection_length,
)

# Capture two hidden-state conditions at two decoder layers.
with vlm.capture("block", layers=[16, 20], positions="last") as positive_cache:
    vlm.generate(positive_inputs, max_new_tokens=1)
with vlm.capture("block", layers=[16, 20], positions="last") as negative_cache:
    vlm.generate(negative_inputs, max_new_tokens=1)

# Directions point from the negative condition toward the positive condition.
direction_16 = mean_difference(positive_cache[16], negative_cache[16], dim=0)
direction_20 = mean_difference(positive_cache[20], negative_cache[20], dim=0)
unit_direction = normalize_direction(direction_16)

# Analyze activation projection and cross-layer direction geometry.
coefficient = projection_coefficient(positive_cache[16], unit_direction)
signed_length = projection_length(positive_cache[16], unit_direction)
component = project(positive_cache[16], unit_direction)
similarity = cosine_similarity(direction_16, direction_20)
theta_degrees = angle(direction_16, direction_20, degrees=True)
```

`mean_difference(positive, negative)` computes the difference between independent sample means; sample counts may differ. It is also equal to the mean of pairwise differences when the two sample sets are aligned and have the same size, so VLMBox exposes only this one aggregate direction operation. It uses the `positive - negative` sign convention and reduces the sample dimension (`dim=0` by default). `normalize_direction` normalizes along the feature dimension (`dim=-1` by default).

`projection_coefficient` returns `dot(hidden, direction) / dot(direction, direction)`. `projection_length` returns the signed scalar projection onto a unit direction, and `project` returns the corresponding vector component. `cosine_similarity` returns cosine similarity, while `angle` returns radians by default or degrees with `degrees=True`. Geometry operations support PyTorch broadcasting and use float32 accumulation for fp16/bf16 inputs. Zero directions are rejected rather than silently producing NaNs.

## Research example: paired-prompt steering

The paired-prompt probe is a small end-to-end example of VLMBox's public research APIs. It captures the last-token decoder activation during prompt prefill for two instructions over the same image, extracts a training-free mean-difference direction, then applies that direction during generation at several intervention scales.

The default synthetic image contains a red square and a blue circle. The positive and negative prompts request `RED` and `BLUE`, while the task prompt asks for the circle's color. This makes the experiment easy to inspect, but it is only an API and causal-intervention sanity probe: one image and one prompt pair do not establish general steering effectiveness.

Run it on a CUDA GPU:

```bash
uv run python examples/paired_prompt_steering.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --scales 0.5,1,2,4
```

Use `--image` to supply another image, `--layer` to select a decoder layer, and `--scope first_forward` to compare one-shot prefill intervention with persistent intervention. Pass `--local-files-only` when the checkpoint is already cached and network access should be prohibited. The JSON result includes baseline and steered text plus the measured prefill activation-delta norm, projection, and cosine similarity. These metrics describe the first prompt-prefill forward pass, not every decode step. Low-precision model execution introduces quantization error, so these measurements should be interpreted with appropriate numerical tolerance.

## Image

Image helpers are architecture-independent and use Pillow. They return new RGB images and do not mutate their inputs:

```python
from PIL import Image
from vlmbox.image import crop_image, place_patch, resize_image, to_rgb

image = to_rgb(Image.open("example.jpg"))
resized = resize_image(image, (672, 756))  # (width, height), LANCZOS by default
patch = crop_image(image, (0, 0, 192, 192))  # (left, top, right, bottom)
composed = place_patch(image, patch, (32, 32))  # (left, top), no implicit resize
```

`crop_image` uses half-open pixel coordinates and rejects empty or out-of-bounds boxes. `place_patch` requires the entire patch to fit inside the base image. An optional same-size `L` or `1` mask blends the patch into the base (`0` keeps the base; `255` replaces it). `to_rgb` follows Pillow conversion semantics and discards an existing alpha channel rather than compositing it.

## Generation

`VLM.generate()` remains a thin pass-through to Transformers. Use `generate_trace()` when a structured record of the generation process is needed:

```python
trace = vlm.generate_trace(
    inputs,
    max_new_tokens=32,
    capture_logits=True,
    capture_hidden_states=True,
    hidden_state_layers=[8, 16, 24],
    decode_text=True,
)

generated_ids = trace.generated_token_ids
first_step_logits = trace.steps[0].logits
prefill_state = trace.steps[0].hidden_states[16]
decode_state = trace.steps[1].hidden_states[16]
print(trace.finish_reasons)
```

`GenerationTrace` stores the returned sequences, prompt length, decoded text (when requested), per-sequence finish reasons, and one `GenerationStep` per generated token. Step zero is the prompt-prefill forward pass; later steps are decode passes. Each step's `token_ids` are the next tokens selected from that pass. Its hidden states describe the prompt or preceding input token(s), not the selected token itself. Hidden-state layer indices are zero-based decoder indices. `decode` denotes later generation calls; with `use_cache=False`, those calls may still process the full prefix.

Logits and hidden states are disabled by default to avoid retaining large GPU tensors. `capture_logits=True` stores raw, unprocessed next-token logits. `capture_hidden_states=True` requires an explicit `hidden_state_layers` selection, preventing accidental retention of every layer. The trace structure is immutable, while tensor payloads remain ordinary PyTorch tensors. Beam-search internal traces are currently rejected because beam histories need explicit alignment.

The Transformers 5.22.1 generation output does not expose a standard finish-reason field. VLMBox infers `eos` only when an unambiguous generated EOS token is present and `length` when the effective explicit length limit is reached without a custom stopping condition; otherwise it reports `unknown`. EOS IDs shared with padding or forced-EOS configuration are treated as ambiguous. This is intentionally conservative.

`generate_trace()` currently supports decoder-only models. Encoder-decoder models are rejected because the source prompt length and decoder sequence length are not interchangeable.

## Repro

Set the RNG seeds for Python, NumPy's legacy global RNG, and PyTorch CPU/CUDA generators:

```python
from vlmbox.repro import set_seed

set_seed(42)  # seed only; leave deterministic backend settings unchanged
set_seed(42, deterministic=True)  # opt in to deterministic PyTorch algorithms
```

`set_seed()` accepts seeds from 0 through `2**32 - 1`. Deterministic mode enables PyTorch deterministic algorithms, enables cuDNN deterministic mode, and disables cuDNN benchmarking. On CUDA systems, an unset `CUBLAS_WORKSPACE_CONFIG` is set to `:4096:8` before CUDA initialization; an existing non-empty value is preserved. If CUDA is already initialized while the variable is unset, the function raises an error and asks you to restart the process. `warn_only=True` changes unsupported deterministic operations from errors to warnings. TF32 settings are not modified.

Models are resolved through ModelScope's default cache. `VLM.from_pretrained()` accepts a ModelScope model ID only; pass `local_files_only=True` to prohibit downloads and require a complete local cache. The runtime uses `AutoProcessor` and `AutoModelForMultimodalLM`, with `dtype="auto"`, `device_map="auto"`, and fixed `trust_remote_code=True`. Only load model repositories whose code you trust.

The generic `TransformersAdapter` has been validated against Qwen2.5-VL-3B, InternVL3.5-2B-HF, and SmolVLM-500M. GPU smoke tests are the required runtime validation path; CPU-only tests are limited to isolated unit tests.

## Development

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/VLMBox"
uv run pytest
uv run ruff check .
```
