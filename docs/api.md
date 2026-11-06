# Public API

The top-level API exports `VLM`, `Renderer`, and `set_seed`. Training, numerical utilities, adapter extension types, and generation result types use explicit module imports:

```python
from vlmbox import VLM, Renderer, set_seed
from vlmbox.trainer import LossFunction, Sample, Trainer, TrainStep
from vlmbox.utils import mean_difference, normalize_direction, project
from vlmbox.model.generation import GenerationTrace, GenerationStep, GenerationTraceError
from vlmbox.model.adapter import ModelAdapter, TransformersAdapter
```

The public rendering interface is `Renderer`; `renderer.layout` and `renderer.intervention` are implementation details. See [research adapters and migration](research-adapters.md) for the boundary between VLMBox and project code.

## Model interface

### Wrap a native model

```python
vlm = VLM(model, processor)
# Optional: adapter=custom_adapter, model_id="org/model", model_path="/checkpoint"
```

VLMBox keeps the original model and processor objects and infers the decoder adapter. An explicit adapter supports other layouts. Wrapping sets tokenizer left padding and preserves model weights, device placement, parameter flags, and training/evaluation modes. The caller owns model loading and freezing when using this constructor. The supported dependency versions still apply.

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

### Canonical decoder layers

All VLMBox layer indices are zero-based decoder block indices: `0` is the first block and `len(vlm.adapter.layers) - 1` is the last. This applies to `adapter.layer`, `capture`, `intervene`, `hidden_state_layers`, and `GenerationStep.hidden_states` keys. Attention and MLP selection use their containing block's index.

A block state is that block's direct output, before any separate final model normalization. Capture and generation traces use this same quantity, including the last block. Embedding states and final-normalized outputs are distinct quantities and are not assigned decoder block indices. Raw native model outputs follow Transformers' own conventions and are an advanced escape hatch, not the VLMBox layer contract.

### Input preparation and execution

```python
vlm.prepare(messages, *, add_generation_prompt=True, chat_template_kwargs=None)
vlm.prepare_differentiable(
    messages,
    *,
    add_generation_prompt=True,
    chat_template_kwargs=None,
)
vlm.forward(inputs, **forward_kwargs)
vlm.generate(inputs, **generation_kwargs)
```

Both preparation methods accept one conversation or a batch of conversations. Set `add_generation_prompt=False` when the conversation already includes an assistant target. `chat_template_kwargs` passes model-specific variables to the native Transformers chat template, for example `{"enable_thinking": False}`. See [preprocessing](preprocessing.md) for message formats, padding, Tensor image constraints, device placement, and gradient behavior.

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

Projection uses at least FP32 arithmetic and preserves the activation's original dtype. Small FP32 directions remain in FP32 during projection even when activations use FP16 or BF16.

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

`GenerationTrace` contains returned sequences, one `GenerationStep` per generated token, finish-reason estimates, and optional decoded text. Hidden-state capture requires explicit zero-based block selection. Temporary hooks record only those blocks' direct outputs; the implementation disables native `output_hidden_states` to avoid embedding offsets and final-normalization differences. Hook removal is guaranteed on success and failure. The first trace step's states equal `capture("block", positions="all")` from that generation.

Logits and hidden states are opt-in. Trace tensors remain on the model device, and retained memory scales with batch size, prompt length, generated tokens, selected blocks, and hidden size. Copying states to CPU after return cannot reduce the generation-time peak. Internal capture supports decoder-only generation with `num_beams=1` and one block forward per generated token; other forward schedules fail explicitly when outputs cannot be aligned.

A step records the forward pass that predicts its token: step zero is prompt prefill, later steps are decode. Its states describe the prompt or preceding token(s), before selecting that step's token. GPU tests verify the same block semantics on Qwen2.5-VL, InternVL, and SmolVLM.

## Numerical utilities

Import these helpers from `vlmbox.utils`. They accumulate in at least FP32 for low-precision inputs and preserve autograd.

- `mean_difference(positive, negative, dim=0)`: mean positive-minus-negative difference along `dim`.
- `normalize_direction(direction, dim=-1, eps=1e-12)`: normalize a direction along a feature dimension.
- `projection_length(hidden, direction, dim=-1, eps=1e-12)`: signed scalar projection onto the unit-normalized direction.
- `project(hidden, direction, dim=-1, eps=1e-12)`: return the vector projection onto `direction`.
- `cosine_similarity(first, second, dim=-1, eps=1e-12)`: compare directions.

Inputs follow PyTorch broadcasting rules where applicable. Tensor pairs must be on the same device, use floating-point dtypes, and have non-zero direction norms.

## YAML-configured image rendering

`Renderer.from_yaml("path/to/image_render.yaml")` loads one complete rendering configuration. Relative font paths are resolved relative to the YAML file. This is the only public image-rendering class.

```python
from PIL import Image
from vlmbox import Renderer

renderer = Renderer.from_yaml("configs/image_render.yaml")
host_image = Image.open("host.png")
rendered_image = renderer.render(host_image, "Execute the instruction in the image.")
```

- `host_image` must be a `PIL.Image.Image`; the host's decoded dimensions are preserved.
- `render()` returns a `PIL.Image.Image`. Set `return_metadata=True` to also receive a plain metadata dictionary.
- Metadata identifies the rendering implementation, font, YAML configuration, and instruction. With an intervention, `intervention_sha256` identifies the current parameter bytes; it changes when the learned parameters change.
- The YAML selects `none`, `overlay_patch`, or `local_additive` and defines the intervention's absolute pixel size and anchor.
- `Renderer` is a PyTorch `nn.Module`. Calling `renderer(host_image, instruction)` returns a differentiable RGB CHW Tensor in `[0, 1]`; `.render(...)` returns PIL.
- Use standard `.parameters()` and `.to(device)` in a custom optimization loop. After each optimizer update, call `.project_()` to enforce the configured bounds. `Trainer` uses the same interface.
- Typography helpers and the underlying intervention implementation remain outside the supported public API.

Save a learned intervention and reuse it on other host images:

```python
renderer.save_intervention("intervention.pt")
restored = Renderer.from_yaml("configs/image_render.yaml")
restored.load_intervention("intervention.pt")
image = restored.render(another_host_image, "Follow this instruction.")
```

Checkpoints contain only weights and intervention settings. Loading requires matching type, pixel size, anchor, and epsilon; host images and typography are independent. Invalid formats, shapes, non-finite values, and values outside the configured bounds are rejected before weights change. A renderer configured with `type: none` has no weights to save or load. Checkpoints are loaded with PyTorch's `weights_only=True`.

## Universal image optimization

Import `Sample`, `Trainer`, `TrainStep`, and the `LossFunction` protocol from `vlmbox.trainer`.

`Sample(image, prompt, system_prompt=None, target=None, image_instruction=None)` describes a native-size PIL host image, the ordinary conversation prompt, and the optional system/assistant text. When training with `Renderer`, every sample must provide `image_instruction`, the text rendered into the image.

`Trainer(vlm, renderer, loss_function, *, learning_rate=1e-2, optimizer_cls=torch.optim.Adam, optimizer_kwargs=None)` optimizes only the trainable intervention configured inside the renderer. The VLM is frozen and placed in evaluation mode during fitting; its weights are not updated. Gradients pass through the frozen VLM to the renderer's patch or perturbation parameters.

```python
from vlmbox.trainer import Trainer

renderer = Renderer.from_yaml("configs/image_render.yaml")
trainer = Trainer(vlm, renderer, loss_function=loss_function)
history = trainer.fit(
    samples,
    max_steps=500,
    batch_size=2,
    shuffle=True,
    seed=2026,
)
```

The `loss_function` argument implements the callable `LossFunction` protocol:

```python
class LossFunction(Protocol):
    def __call__(
        self,
        outputs: Any,
        inputs: Mapping[str, Any],
        samples: Sequence[Sample],
        /,
    ) -> Tensor: ...
```

| Argument | Meaning |
|---|---|
| `outputs` | The native result of the current model forward; for example logits or requested hidden states |
| `inputs` | The prepared model inputs, including token IDs, padding masks, and model-specific image tensors/metadata |
| `samples` | The matching unprocessed samples, including prompts, target strings, and sample-specific reference lookup |

All three arguments use the same batch order. `inputs` and `samples` serve different purposes: token alignment uses prepared tensors, while target/reference selection uses original sample information. They are passed positionally, so implementations may choose their own parameter names. Ordinary functions, closures, and callable objects work without inheritance. The protocol is a static typing contract; the trainer validates the returned Tensor and gradient path at runtime.

The callable returns one finite scalar Tensor to minimize, with gradients connected to the renderer intervention. `trainer.loss_function` stores the callable; `TrainStep.loss` records its scalar value. This interface computes a loss after a single prepared forward. Losses that need to control preprocessing, install capture hooks around the forward, or arrange multiple forwards belong in a custom research loop using the VLM operations. `fit()` runs exactly `max_steps` updates and returns a tuple of `TrainStep(step, loss, gradient_norm)` records. Each model parameter's original `requires_grad` flag and each submodule's training/evaluation mode are restored on success or failure.

The runnable image optimization example writes a preview image, JSON diagnostics, and a `.pt` intervention checkpoint. Pass `--intervention saved.pt` to start from learned weights. This starts a fresh optimizer; the checkpoint does not include optimizer state or dataset position.

The implementation-specific PyTorch parameter classes are not part of the package-level public API. Select the intervention through YAML instead of constructing or managing those classes directly.

## Visual input rendering (VIRP)

VIRP's typography layout, band composition, and differentiable tensor composition are internal implementation details behind `Renderer`. The package-level API exposes neither their helper classes nor their helper functions. Host-relative coordinate helpers are also internal. Ordinary image conversion, resizing, cropping, and PIL patch composition use Pillow directly.

See the [Visual Input Rendering Protocol](protocols/visual-input-rendering.md). The package does not bundle proprietary fonts or resize host images in these APIs.

## Reproducibility

```python
set_seed(seed, *, deterministic=False, warn_only=False)
```

Seeds Python, NumPy, and PyTorch. With `deterministic=True`, enables PyTorch deterministic algorithms and configures cuDNN deterministic behavior. When CUDA is available, the cuBLAS workspace setting must be established before CUDA initialization; see the function's runtime error if it is called too late.
