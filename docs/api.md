# API reference

For complete workflows, see [usage](usage.md). The package root exports `VLM`, `Renderer`, and `set_seed`.

```python
from vlmbox import VLM, Renderer, set_seed
from vlmbox.trainer import Sample, Trainer, TrainStep, LossFunction
from vlmbox.model.adapter import ModelAdapter, TransformersAdapter
from vlmbox.model.generation import GenerationTrace, GenerationStep, GenerationTraceError
from vlmbox.utils import (
    mean_difference,
    normalize_direction,
    project,
    projection_length,
    cosine_similarity,
)
```

## VLM

| Interface | Contract |
|---|---|
| `VLM(model, processor, *, adapter=None, model_id=None, model_path=None)` | Wrap native objects; preserve model state and set tokenizer left padding |
| `VLM.from_pretrained(model_id, *, local_files_only=False)` | Load a ModelScope `namespace/model` ID with automatic dtype/device placement |
| `.prepare(messages, *, add_generation_prompt=True, chat_template_kwargs=None)` | Native preparation of a conversation or batch |
| `.prepare_differentiable(messages, *, add_generation_prompt=True, chat_template_kwargs=None)` | Tensor-image preparation retaining gradients; finite RGB CHW floats in `[0, 1]` |
| `.forward(inputs, **forward_kwargs)` | Native model forward with autograd |
| `.generate(inputs, **generation_kwargs)` | Native Transformers generation |
| `.capture(component, layers, positions="last")` | Context returning first-invocation activations, accessed as `capture[layer]` |
| `.intervene(component, layer, operation, value=None, scale=1.0, positions="last", scope="every_forward")` | Temporary activation-edit context |
| `.generate_trace(inputs, *, capture_logits=False, capture_hidden_states=False, hidden_state_layers=None, decode_text=False, **generation_kwargs)` | Structured generation with opt-in logits and selected block states |

Properties: `.model`, `.processor`, `.adapter`, `.input_device`, `.model_id`, `.model_path`. `ModelAdapter` supplies `.layers`, `.layer(index)`, `.attention(index)`, and `.mlp(index)`.

Layers are zero-based decoder blocks. States are direct block outputs before separate final normalization, consistently across capture and traces. Native embedding/final-normalized states follow native model conventions. `component` is `block`, `attention`, or `mlp`; `positions` is `all`, `last`, an integer, or a sequence of integers.

| Operation | Effect on selected activations |
|---|---|
| `add` | `h + scale * value` |
| `replace` | Broadcast `value`; scale must be 1 |
| `project_out` | Remove `scale` times the vector projection onto `value`, using at least FP32 arithmetic |
| `zero` | Zero activations; omit `value`, and scale must be 1 |

Scope is `first_forward` or `every_forward`. Hooks run in context-registration order and are removed on exit. Captures retain gradients unless the caller disables autograd.

`GenerationTrace` exposes `prompt_length`, `sequences`, `prompt_token_ids`, `generated_token_ids`, `steps`, `finish_reasons`, and optional `texts`. Each `GenerationStep` contains `index`, `phase`, `token_ids`, optional raw `logits`, and a layer-to-Tensor `hidden_states` mapping. States describe the forward **predicting** that step's token. Hidden-state capture requires explicit layer selection; internal capture supports decoder-only, single-beam generation with one block forward per generated token. Finish reasons are `eos`, `length`, or `unknown` estimates.

Differentiable preprocessing uses contiguous CPU float32 images with rescaling disabled, then transfers prepared inputs to `.input_device` without detaching. The native processor owns resize, crop, patch layout, and normalization. Exact results depend on the checkpoint and installed dependencies.

## Renderer

| Interface | Contract |
|---|---|
| `Renderer.from_yaml(path, *, device=None)` | Strict schema validation; font paths are relative to the YAML |
| `renderer(host_image, instruction, *, draw_text=True)` | Differentiable RGB CHW Tensor in `[0, 1]` |
| `.render(host_image, instruction, *, draw_text=True, return_metadata=False)` | PIL image, or `(image, metadata)` |
| `.parameters()`, `.to(...)`, `.zero_grad(...)` | Standard PyTorch module operations |
| `.project_()` | Enforce intervention bounds after an optimizer update |
| `.save_intervention(path)`, `.load_intervention(path)` | Save/restore weights with matching type, size, anchor, and epsilon |

Inputs are native-size PIL hosts. YAML chooses `none`, `overlay_patch`, or `local_additive`. Checkpoints exclude optimizer state and reject invalid shapes, non-finite values, or out-of-bounds weights before mutation. Metadata records geometry, font/config/instruction hashes, and intervention weight hashes. See the [YAML](../examples/image_render.yaml) and [rendering protocol](protocols/visual-input-rendering.md).

## Trainer

```text
Sample(image, prompt, system_prompt=None, target=None, image_instruction=None)
Trainer(vlm, renderer, loss_function, *, learning_rate=1e-2,
        optimizer_cls=torch.optim.Adam, optimizer_kwargs=None)
trainer.fit(samples, *, max_steps, batch_size=1, shuffle=True,
            seed=2026, forward_kwargs=None) -> tuple[TrainStep, ...]
```

Every training sample needs `image_instruction`. All samples must either provide `target` text or omit it; provided targets disable the generation prompt. `fit()` cycles through samples for exactly `max_steps` updates, freezes the model, and restores every original parameter flag and submodule training mode on success or failure. `TrainStep` stores `step`, scalar `loss`, and `gradient_norm`.

`LossFunction` is a structural callable protocol with positional signature `(outputs, inputs, samples) -> Tensor`. Arguments share batch order: native model results, prepared model inputs, and original samples. Return a finite scalar minimization loss connected to renderer parameters. Single-forward losses use `Trainer`; hooks or multiple forwards use a [custom loop](usage.md#custom-differentiable-loops).

## Utilities

| Function | Meaning |
|---|---|
| `mean_difference(positive, negative, dim=0)` | Mean positive-minus-negative difference; sample counts may differ |
| `normalize_direction(direction, dim=-1, eps=1e-12)` | Unit directions; reject zero/non-finite norms |
| `project(hidden, direction, dim=-1, eps=1e-12)` | Vector projection |
| `projection_length(hidden, direction, dim=-1, eps=1e-12)` | Signed projection onto the unit direction |
| `cosine_similarity(first, second, dim=-1, eps=1e-12)` | Cosine with broadcasting and nonzero-norm checks |
| `set_seed(seed, *, deterministic=False, warn_only=False)` | Seed Python, NumPy, and PyTorch; optionally enable deterministic algorithms |

Numerical helpers preserve autograd and use at least FP32 for half-precision inputs. Tensor pairs must share a device. Seeds range from 0 through `2**32 - 1`. For deterministic CUDA, call `set_seed` before CUDA initialization when `CUBLAS_WORKSPACE_CONFIG` is unset; `warn_only` requires `deterministic=True`.
