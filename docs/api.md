# Public API

The symbols below are exported from the top-level `vlmbox` package unless otherwise noted.

## Model interface

### `VLM.from_pretrained`

```python
VLM.from_pretrained(model_id: str, *, local_files_only: bool = False) -> VLM
```

Load a ModelScope checkpoint through Transformers. `model_id` must use the `namespace/model` form. Set `local_files_only=True` to require a complete local ModelScope cache. Model dtype and device placement currently use the internal `auto` defaults. For InternVL checkpoints that omit the top-level `tie_word_embeddings` field, VLMBox follows the nested text model setting when it specifies untied embeddings; explicit checkpoint settings are preserved.

Useful properties:

- `vlm.model`: underlying Transformers model.
- `vlm.processor`: native multimodal processor.
- `vlm.adapter`: decoder component adapter.
- `vlm.input_device`: device used for model inputs.

### Input preparation and execution

```python
vlm.prepare(messages, *, chat_template_kwargs=None)
vlm.prepare_differentiable(
    messages,
    *,
    add_generation_prompt=True,
    chat_template_kwargs=None,
)
vlm.forward(inputs, **forward_kwargs)
vlm.generate(inputs, **generation_kwargs)
```

Both preparation methods accept one conversation or a batch of conversations. `chat_template_kwargs` passes model-specific variables to the native Transformers chat template, for example `{"enable_thinking": False}`. See [preprocessing](preprocessing.md) for message formats, padding, Tensor image constraints, device placement, and gradient behavior.

`forward()` is autograd-enabled. `generate()` delegates decoding to Transformers and is inference-oriented.

### Activation capture and intervention

```python
with vlm.capture(component="block", layers=12, positions="last") as cache:
    outputs = vlm.generate(inputs, max_new_tokens=1)
state = cache[12]

with vlm.intervene(
    component="block",
    layer=12,
    operation="add",
    value=direction,
    scale=1.0,
    positions="last",
    scope="every_forward",
):
    outputs = vlm.generate(inputs, max_new_tokens=16)
```

`component` is `block`, `attention`, or `mlp`. `positions` accepts `all`, `last`, an integer, or a sequence of token indices. Capture records the first invocation of each selected module.

Intervention operations:

- `add`: add `scale * value`.
- `replace`: replace selected activations with `value`.
- `project_out`: remove the component of the activation along `value`, scaled by `scale`.
- `zero`: set selected activations to zero; `value` is not required.

`scope="first_forward"` applies the intervention once per selected module; `scope="every_forward"` applies it on every invocation. The latter is persistent across autoregressive decoding.

### Generation traces

```python
trace = vlm.generate_trace(
    inputs,
    capture_logits=True,
    capture_hidden_states=True,
    hidden_state_layers=[8, 16, 24],
    decode_text=True,
    max_new_tokens=32,
)
```

`GenerationTrace` contains returned sequences, one `GenerationStep` per generated token, finish-reason estimates, and optional decoded text. Hidden-state capture requires an explicit layer selection. VLMBox passes those zero-based indices to Transformers' native `output_hidden_states` mechanism. Model implementations with native layer-list selection return tensors only for requested layers (unselected entries are `None`). A model that treats this option as a boolean may still return every layer; VLMBox continues to filter the requested layers, but that model will not receive the memory benefit of native selective capture. The GPU integration test verifies native selection on Qwen2.5-VL-3B-Instruct, InternVL3.5-2B-HF, and SmolVLM-500M-Instruct. Logits and hidden states are opt-in because their retained memory scales with batch size, prompt length, generated tokens, selected layers, and hidden size. Internal trace capture currently supports decoder-only generation and `num_beams=1`.

Trace tensors remain on the model's device. CPU offload is not automatic: copying tensors after `generate()` returns can reduce the VRAM retained by a trace, but cannot reduce the generation-time peak because Transformers has already accumulated the per-step outputs. Streaming those outputs to CPU during decoding would require intercepting generation forwards or customizing the generation loop, adding transfer overhead and version-sensitive logic.

A step records the forward pass that predicts its token: step zero is prompt prefill, later steps are decode. Its hidden states describe the input to that forward pass, not the token selected from its logits.

## Representation utilities

- `mean_difference(positive, negative, dim=0)`: mean positive-minus-negative difference along `dim`.
- `normalize_direction(direction, dim=-1, eps=1e-12)`: normalize a direction along a feature dimension.
- `projection_coefficient(hidden, direction, dim=-1, eps=1e-12, keepdim=False)`: return `dot(hidden, direction) / dot(direction, direction)`.
- `projection_length(hidden, direction, dim=-1, eps=1e-12)`: signed scalar projection onto the unit-normalized direction.
- `project(hidden, direction, dim=-1, eps=1e-12)`: return the vector projection onto `direction`.
- `cosine_similarity(first, second, dim=-1, eps=1e-12)` and `angle(first, second, dim=-1, degrees=False, eps=1e-12)`: compare directions.

Inputs follow PyTorch broadcasting rules where applicable. Tensor pairs must be on the same device, use floating-point dtypes, and have non-zero direction norms.

## Image utilities

- `to_rgb(image)`: convert a PIL image to RGB.
- `resize_image(image, size, resample=Image.Resampling.LANCZOS)`: resize a PIL image; `size` is `(width, height)`.
- `crop_image(image, box)`: crop a PIL image using `(left, top, right, bottom)`.
- `place_patch(image, patch, position, mask=None)`: place a PIL patch at an `(x, y)` position, optionally using a mask.

## YAML-configured image rendering

`ImageRenderer.from_yaml("path/to/image_render.yaml")` loads one complete rendering configuration. Relative font paths are resolved relative to the YAML file. This is the only public image-rendering class.

```python
from PIL import Image
from vlmbox import ImageRenderer

renderer = ImageRenderer.from_yaml("configs/image_render.yaml")
host_image = Image.open("host.png")
rendered_image = renderer.render(host_image, "Execute the instruction in the image.")
```

- `host_image` must be a `PIL.Image.Image`; the host's decoded dimensions are preserved.
- `render()` returns a `PIL.Image.Image`. Set `return_metadata=True` to also receive a plain metadata dictionary.
- The YAML selects `none`, `overlay_patch`, or `local_additive` and defines the intervention's absolute pixel size and anchor.
- Tensor conversion, differentiable rendering, intervention parameters, and parameter projection are internal implementation details. Users do not call `forward()`, `render_tensor()`, `parameters()`, or `project_()`.

## Universal image optimization

`ImageSample(image, prompt, system_prompt=None, target=None, image_instruction=None)` describes a native-size PIL host image, the ordinary conversation prompt, and the optional system/assistant text. When training with `ImageRenderer`, every sample must provide `image_instruction`, the text rendered into the image.

`ImageTrainer(vlm, renderer, objective, *, learning_rate=1e-2, optimizer_cls=torch.optim.Adam, optimizer_kwargs=None)` optimizes only the trainable intervention configured inside the renderer. The VLM is frozen and placed in evaluation mode during fitting; its weights are not updated. Gradients pass through the frozen VLM to the renderer's patch or perturbation parameters.

```python
renderer = ImageRenderer.from_yaml("configs/image_render.yaml")
trainer = ImageTrainer(vlm, renderer, objective)
history = trainer.fit(
    samples,
    max_steps=500,
    batch_size=2,
    shuffle=True,
    seed=2026,
)
```

The objective receives `(model_outputs, model_inputs, samples)` and must return a scalar differentiable Tensor. `fit()` runs exactly `max_steps` updates and returns a tuple of `TrainingStep(step, loss, gradient_norm)` records.

The implementation-specific PyTorch parameter classes are not part of the package-level public API. Select the intervention through YAML instead of constructing or managing those classes directly.

## Visual input rendering (VIRP)

VIRP's typography layout, band composition, and differentiable tensor composition are internal implementation details behind `ImageRenderer`. The package-level API exposes neither their helper classes nor their helper functions. The general-purpose `anchored_bbox((host_width, host_height), (patch_width, patch_height), anchor)` geometry helper remains available.

See the [Visual Input Rendering Protocol](protocols/visual-input-rendering.md). The package does not bundle proprietary fonts or resize host images in these APIs.

## Reproducibility

```python
set_seed(seed, *, deterministic=False, warn_only=False)
```

Seeds Python, NumPy, and PyTorch. With `deterministic=True`, enables PyTorch deterministic algorithms and configures cuDNN deterministic behavior. When CUDA is available, the cuBLAS workspace setting must be established before CUDA initialization; see the function's runtime error if it is called too late.
