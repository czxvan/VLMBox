# VLMBox

A lightweight Python research toolbox for vision-language models (VLMs).

VLMBox provides reusable model-inspection and universal-image-optimization primitives. It is intentionally not a general-purpose fine-tuning framework: research-specific objectives belong in the research project that defines them.

## Scope

- **Model** — load ModelScope checkpoints through Transformers and prepare native model inputs.
- **Adapter** — resolve common language-decoder components without coupling research code to one model family.
- **Activation** — capture and temporarily intervene on decoder blocks, attention, and MLP outputs.
- **Representation** — training-free direction extraction and representation geometry.
- **Image** — architecture-independent RGB conversion, resizing, cropping, and patch placement.
- **Training** — optimize one universal image perturbation or patch with a task-supplied objective.
- **Generation** — structured generation results and opt-in internal traces.
- **Repro** — seed control and optional deterministic execution.

## Installation

### Install from PyPI

```bash
python -m pip install vlmbox
```

The PyPI package declares a compatible PyTorch version range but does not force a particular CUDA build. If you need a specific accelerator build, install the matching PyTorch version for your platform first, then install VLMBox. The optional `examples` extra installs torchvision for the runnable image-optimization example:

```bash
python -m pip install "vlmbox[examples]"
```

### Development environment

The repository's `uv.lock` intentionally pins CUDA 13.0 PyTorch wheels for the lab's RTX 3090 and RTX 4090 machines. The supported Linux research machines require an NVIDIA driver version 580 or newer. Use a machine-local environment rather than placing `.venv` on the shared workspace:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/VLMBox"
uv sync --locked --extra dev --extra examples
```

The Tsinghua PyPI mirror and PyTorch CUDA 13.0 index are declared in `pyproject.toml`; `uv.lock` pins the resolved wheels. Use the same commands on each lab machine. Do not mutate this environment with a separate CPU-only PyTorch installation, pip, or Conda.

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

Capture decoder outputs without modifying the model:

```python
with vlm.capture("block", layers=[8, 16, 24], positions="last") as cache:
    vlm.generate(inputs, max_new_tokens=32)

last_layer_state = cache[24]  # [batch, 1, hidden_size]
```

Apply a temporary intervention to one decoder layer:

```python
with vlm.intervene(
    "block",
    layer=16,
    operation="add",
    value=direction,
    positions="last",
    scope="every_forward",
):
    output_ids = vlm.generate(inputs, max_new_tokens=32)
```

Supported components are `block`, `attention`, and `mlp`. Position selectors are `last`, `all`, or explicit token indices. `capture` records the first invocation of each selected module.

Intervention operations:

- `add`: add a scaled direction to the selected activation.
- `replace`: replace the selected activation.
- `project_out`: remove the projection onto a direction.
- `zero`: set the selected activation to zero.

Values must broadcast to the selected activation shape. `scale` applies to `add` and `project_out`. `first_forward` intervenes once per target layer; `every_forward` intervenes on every invocation, including decoding. Nested hooks run in registration order and are removed when the context exits. Captured tensors retain their autograd graph; detach them when graph retention is unnecessary.

## Representation

Representation helpers are pure PyTorch operations; they do not train a classifier.

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

with vlm.capture("block", layers=[16, 20], positions="last") as positive_cache:
    vlm.generate(positive_inputs, max_new_tokens=1)
with vlm.capture("block", layers=[16, 20], positions="last") as negative_cache:
    vlm.generate(negative_inputs, max_new_tokens=1)

direction_16 = mean_difference(positive_cache[16], negative_cache[16], dim=0)
direction_20 = mean_difference(positive_cache[20], negative_cache[20], dim=0)
unit_direction = normalize_direction(direction_16)

coefficient = projection_coefficient(positive_cache[16], unit_direction)
signed_length = projection_length(positive_cache[16], unit_direction)
component = project(positive_cache[16], unit_direction)
similarity = cosine_similarity(direction_16, direction_20)
theta_degrees = angle(direction_16, direction_20, degrees=True)
```

`mean_difference(positive, negative)` computes the difference between independent sample means; sample counts may differ. It uses the `positive - negative` sign convention and reduces the sample dimension (`dim=0` by default). When two aligned sample sets have the same size, this equals the mean of their pairwise differences. `normalize_direction` normalizes along the feature dimension (`dim=-1` by default).

`projection_coefficient` returns `dot(hidden, direction) / dot(direction, direction)`. `projection_length` returns the signed scalar projection onto a unit direction, and `project` returns the corresponding vector component. `cosine_similarity` and `angle` compute vector similarity and angular distance. Geometry operations support PyTorch broadcasting and use float32 accumulation for fp16/bf16 inputs. Zero directions are rejected.

### Paired-prompt steering example

The example demonstrates a complete training-free workflow: capture two prompt-conditioned activations over the same image, extract a mean-difference direction, then apply it during generation at several scales. Its synthetic red-square/blue-circle task is an API sanity probe, not evidence of general steering effectiveness.

```bash
uv run python examples/paired_prompt_steering.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --scales 0.5,1,2,4
```

Use `--image` to provide an image, `--layer` to select a decoder layer, and `--scope first_forward` to compare prefill-only and persistent intervention. Use `--local-files-only` when the checkpoint is already cached.

## Image

Image helpers are architecture-independent and use Pillow. They return new RGB images without mutating their inputs:

```python
from PIL import Image
from vlmbox.image import crop_image, place_patch, resize_image, to_rgb

image = to_rgb(Image.open("example.jpg"))
resized = resize_image(image, (672, 756))  # (width, height), LANCZOS by default
patch = crop_image(image, (0, 0, 192, 192))  # (left, top, right, bottom)
composed = place_patch(image, patch, (32, 32))  # (left, top), no implicit resize
```

`crop_image` uses half-open pixel coordinates and rejects empty or out-of-bounds boxes. `place_patch` requires the patch to fit inside the base image. An optional same-size `L` or `1` mask blends the patch into the base. `to_rgb` follows Pillow conversion semantics and discards an existing alpha channel rather than compositing it.

## Universal image training

VLMBox currently supports **universal** image optimization only: one trainable image parameter is shared across all samples in a run. It does not implement per-image optimization.

- `UniversalPerturbation` learns one full-canvas RGB perturbation shared by every sample and constrained by an L-infinity bound `epsilon`.
- `UniversalPatch` learns one RGB patch shared by every sample and places it at one fixed canvas location.
- `ImageTrainer` manages batching, model freezing, forward/backward passes, optimizer updates, and projection back into the parameter's feasible range.

### Objective and Trainer

`ImageTrainer` is task-agnostic. Its `objective` argument accepts either a function or any callable object with the signature:

```python
def objective(outputs, inputs, samples):
    # Return one differentiable scalar Tensor.
    ...
```

The objective must return one differentiable scalar tensor. It can use logits, hidden states, or other model outputs. Pass model-specific forward options through `forward_kwargs`, for example `{"output_hidden_states": True}` for a hidden-state objective.

A typical setup looks like this:

```python
from vlmbox import ImageTrainer, UniversalPatch

patch = UniversalPatch(
    canvas_size=(224, 224),
    size=(48, 48),
    position="bottom_right",
)
trainer = ImageTrainer(vlm, patch, objective, learning_rate=1e-3)
history = trainer.fit(samples, max_steps=500, batch_size=2, seed=42)
```

For hidden-state objectives, pass `forward_kwargs={"output_hidden_states": True}` to `fit()`.

A new loss does **not** require a new Trainer. Implement the objective in the research project and pass it to `ImageTrainer`. VLMBox intentionally does not bundle task-specific losses or method-specific Trainers. A custom Trainer is only warranted when a project needs a genuinely different optimization loop, not merely a different loss.

### Data and preprocessing

Each `ImageSample` contains an RGB float image tensor in `[0, 1]`, a user prompt, and optional system prompt or teacher-forced target text. All images in one run must share the same `(height, width)`. Either every sample provides target text or none does. When target text is present, it is appended as an assistant turn; the objective is responsible for selecting the target-token positions used by its loss.

`VLM.prepare_differentiable()` accepts Tensor image content in one conversation or a batch of conversations. It uses the model's native Transformers processor, with rescaling disabled because inputs are already in `[0, 1]`. Native resizing, cropping, patchification, splitting, normalization, and padding follow each model processor's own configuration; VLMBox does not override these settings.

Image preprocessing runs on CPU so the configured native resampling path is preserved. The processed tensors are then moved to the VLM input device without detaching the autograd graph. The recommended setup keeps source images and universal image parameters on CPU while the VLM runs on CUDA. `VLM.forward()` is autograd-enabled; `generate()` is inference-oriented.

During `fit()`, the Trainer freezes VLM parameters and switches the model to evaluation mode, then restores their original `requires_grad` flags and training mode. Only the universal image parameter is optimized. The default optimizer is Adam; `optimizer_cls` and `optimizer_kwargs` can be supplied when constructing the Trainer.

### Runnable example

This example uses teacher-forced assistant-target cross-entropy and verifies that the prompt-only token sequence is an exact prefix of the prompt-plus-target sequence. It prepares a fixed square canvas before the model's native processor runs.

```bash
uv run python examples/universal_image_optimization.py \
  --image-dir ./images \
  --parameterization patch \
  --steps 100 \
  --batch-size 2
```

Use `--parameterization perturbation` for a universal L-infinity-bounded perturbation, or `patch` for a fixed-position patch. The default model is `HuggingFaceTB/SmolVLM-500M-Instruct`; pass `--model-id` to select another supported model. The optimized parameter and step history are saved to the requested `.pt` output.

## Generation

`VLM.generate()` is a thin pass-through to Transformers. Use `generate_trace()` when a structured record of generation is needed:

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

`GenerationTrace` stores generated sequences, prompt length, optional decoded text, per-sequence finish reasons, and one `GenerationStep` per generated token. Step zero is prompt prefill; later steps are autoregressive decode. A step's `token_ids` are the tokens selected from that forward pass, while its hidden states describe the prompt or preceding input tokens.

Logits and hidden states are disabled by default to avoid retaining large GPU tensors. `capture_hidden_states=True` requires an explicit `hidden_state_layers` selection. Hidden-state indices are zero-based decoder-layer indices. Beam-search internal traces are currently rejected because beam histories need explicit alignment. `generate_trace()` currently supports decoder-only models.

Transformers does not expose a standard finish-reason field. VLMBox reports `eos` or `length` only when the returned sequence makes the reason unambiguous; otherwise it reports `unknown`.

## Reproducibility

Set RNG seeds for Python, NumPy's legacy global RNG, and PyTorch CPU/CUDA generators:

```python
from vlmbox.repro import set_seed

set_seed(42)  # seed only; leave deterministic backend settings unchanged
set_seed(42, deterministic=True)  # opt in to deterministic PyTorch algorithms
```

`set_seed()` accepts seeds from 0 through `2**32 - 1`. Deterministic mode enables PyTorch deterministic algorithms, enables cuDNN deterministic mode, and disables cuDNN benchmarking. On CUDA systems, an unset `CUBLAS_WORKSPACE_CONFIG` is set to `:4096:8` before CUDA initialization; an existing non-empty value is preserved. If CUDA is already initialized while the variable is unset, the function raises an error and asks you to restart the process. `warn_only=True` changes unsupported deterministic operations from errors to warnings. TF32 settings are not modified.

## Model loading and compatibility

Models are resolved through ModelScope's default cache. `VLM.from_pretrained()` accepts a ModelScope model ID; pass `local_files_only=True` to prohibit downloads and require a complete local cache. VLMBox uses `AutoProcessor` and `AutoModelForMultimodalLM`, with `dtype="auto"`, `device_map="auto"`, and fixed `trust_remote_code=True`. Only load repositories whose code you trust.

The generic `TransformersAdapter` has GPU integration coverage for Qwen2.5-VL-3B, InternVL3.5-2B-HF, and SmolVLM-500M. Other models may work when their language-decoder module layout matches the adapter's supported patterns.

## Development

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/VLMBox"
uv run ruff check .
uv run pytest -q
```

Run real-model integration tests when the three checkpoints are already present in the ModelScope cache:

```bash
VLMBOX_RUN_GPU_INTEGRATION=1 uv run pytest tests/integration -q
```

Build the wheel and source distribution with:

```bash
uv build
```

## License

VLMBox is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

Maintainers can find release and Trusted Publishing instructions in the repository's `RELEASING.md`.