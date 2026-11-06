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

## Visual input rendering (VIRP)

- `InstructionStyle(font_path, font_size=20, margin=20, line_spacing=None, ...)`: fixed font settings with an **external** font; the default spacing equals `int(font_size * 0.2)`.
- `render_instruction_band(text, width, style, draw_text=True)`: RGB band with dynamic height and fully wrapped text. `draw_text=False` produces a geometrically matched blank control.
- `compose_instruction_image(host, text, style, position="top", draw_text=True)`: PIL composed image with `instruction_bbox`, `host_bbox`, line list, font hash, and `metadata()`.
- `compose_instruction_tensor(host, band, position="top")`: differentiable CHW image concatenation after applying a local tensor intervention.
- `anchored_bbox((host_width, host_height), (patch_width, patch_height), anchor)`: half-open host-local bbox without margins; supports five named anchors and explicit `(x, y)`.

See the [Visual Input Rendering Protocol](protocols/visual-input-rendering.md).
The package does not bundle proprietary fonts or perform any host-image resizing in these APIs.

## Universal image optimization

`UniversalPerturbation(canvas_size, epsilon, *, init="zeros", seed=2026, device=None, dtype=torch.float32)` preserves the original fixed-canvas full-image additive behavior.

`UniversalPatch(canvas_size, size, *, position="top_left", init="random", seed=2026, device=None, dtype=torch.float32)` creates a shared overlay patch. `canvas_size=None` permits the same fixed-size patch to be applied to host images with different native dimensions.

`UniversalLocalPerturbation(size, epsilon, *, position="bottom_right", init="zeros", seed=2026, device=None, dtype=torch.float32)` creates one shared, bounded local additive perturbation, likewise usable with native-size images.

Sizes in trainable tensor APIs use `(height, width)`. Explicit positions use `(x, y)`. Named positions are `top_left`, `top_right`, `bottom_left`, `bottom_right`, and `center`.

All three expose a trainable `.parameter`, `apply(images)`, and `project_()`. Image parameters accept either a CHW tensor or BCHW batch. Fixed-canvas parameters enforce one canvas size. Native-size parameters accept varying image sizes and `ImageTrainer` applies them per sample, letting the model's native processor perform batching.


`ImageSample(image, prompt, system_prompt=None, target=None)` describes one RGB float Tensor image in `[0, 1]` and its conversation. `ImageTrainer(vlm, image_parameter, objective, *, learning_rate=1e-2, optimizer_cls=torch.optim.Adam, optimizer_kwargs=None)` optimizes the shared parameter. The objective receives `(model_outputs, model_inputs, samples)` and must return a scalar differentiable Tensor.

```python
history = trainer.fit(
    samples,
    max_steps=500,
    batch_size=2,
    shuffle=True,
    seed=2026,
)
```

`fit()` runs exactly `max_steps` updates and returns a tuple of `TrainingStep(step, loss, gradient_norm)` records. The model is frozen and placed in evaluation mode during fitting; only the universal image parameter is optimized.

## Reproducibility

```python
set_seed(seed, *, deterministic=False, warn_only=False)
```

Seeds Python, NumPy, and PyTorch. With `deterministic=True`, enables PyTorch deterministic algorithms and configures cuDNN deterministic behavior. When CUDA is available, the cuBLAS workspace setting must be established before CUDA initialization; see the function's runtime error if it is called too late.
