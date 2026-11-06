# Research adapters and API migration

VLMBox defines shared model operations and a single zero-based decoder-block contract. Research projects own messages, losses, token masks, metrics, and result formats. The package handles preprocessing, device transfer, model layout differences, hooks, and trace alignment. New project APIs use the VLMBox convention directly.

## Needs grounded in current research code

| Research requirement | VLMBox operation | Project responsibility |
|---|---|---|
| Optional custom loading or unsupported native operations | `VLM(model, processor, adapter=...)`, `model`, `processor` | Use the escape hatch only for needs beyond shared VLMBox operations |
| Differentiate through native image preprocessing | `prepare_differentiable` and `forward` | Assemble conversations and choose the loss |
| Probe an open assistant response prefix | `add_generation_prompt=False`, `chat_template_kwargs={"continue_final_message": True}` | Construct the assistant prefix and select its meaning |
| Capture decoder block outputs | `capture("block", layers=..., positions="last")` | Choose blocks and token positions using the canonical zero-based convention |
| Prompt-only or persistent additive steering | `intervene(..., scope="first_forward" / "every_forward")` | Select direction, magnitude, and experiment condition |
| Remove a direction component | `intervene(..., operation="project_out")` | Select the research direction |
| Analyze a generated trajectory | `generate_trace` | Compute token probabilities, EOS metrics, latency, and aggregate results |
| Apply a special visual-token mask or release after N forwards | `adapter.layer(index)` and native PyTorch hooks | Implement the experiment-specific selection and schedule |

`first_forward` means the module's first invocation in the context. A project needing a different prefill predicate or a finite intervention horizon should define it explicitly with a native hook.

## Thin adapter examples

`VLM.from_pretrained(...)` is the standard loading entry point. For an optional custom loading path, wrap existing objects with:

```python
from vlmbox import VLM

runtime = VLM(model, processor)
model.eval().requires_grad_(False)
inputs = runtime.prepare_differentiable(
    conversations,
    add_generation_prompt=False,
    chat_template_kwargs={"continue_final_message": True},
)
```

Freezing model weights still permits gradients from a scalar loss back to input image tensors. VLMBox's constructor preserves model state and sets tokenizer left padding for batching.

For a decoder-block probe, use zero-based indices directly:

```python
with runtime.capture("block", layers=research_layers, positions="last") as states:
    outputs = runtime.forward(inputs, use_cache=False)
# states[index] is [batch, 1, hidden]; it retains gradients.
```

`capture()` and `generate_trace()` both return direct block outputs before separate final model normalization. Native `hidden_states` tuples can contain embedding offsets or a final-normalized last entry; VLMBox traces do not depend on that layout. Embeddings and final-normalized representations are separate quantities, accessed explicitly through advanced native operations when needed.

For per-sample prompt-end steering:

```python
# delta is [batch, hidden]; insert the token axis to match [batch, 1, hidden].
with runtime.intervene(
    "block",
    layer=research_layer,
    operation="add",
    value=delta[:, None, :],
    positions="last",
    scope="first_forward",
):
    sequences = runtime.generate(inputs, max_new_tokens=32)
```

The adapter also owns any expansion needed for beam or returned-sequence batches. Native model and processor objects remain available for operations outside the library's bounded API.

## Current caller status

LoopPatch already uses `VLM` preparation, capture, intervention, generation traces, and `adapter.layer/layers`. These operations remain available. Its project-specific visual-token interventions and trajectory metrics stay in LoopPatch.

HijackPatch currently has a standalone adapter, including Transformers 4.57-specific preprocessing calls and one-based layer conventions. This cleanup supplies a simple wrapping path and tests its required prefix/gradient behavior, but does not migrate HijackPatch or claim compatibility with its old Transformers runtime. A migration needs the supported Python/PyTorch/Transformers environment and explicit regression checks for padding, prefixes, final normalization, losses, and generation metadata.

## Import migration

The package root exports only `VLM`, `Renderer`, and `set_seed`.

VLMBox owns layer numbering. During migration, convert one-based stored block numbers to zero-based values once at the artifact/configuration boundary. Existing legacy adapters may temporarily perform that conversion, but new experiment interfaces should not invent another convention. Recompute old trajectory caches whose final-layer values came from native final-normalized outputs: those values differ from the canonical raw-block state.

`ImageRenderer` is renamed to `Renderer`, available from `vlmbox` and `vlmbox.renderer`. The YAML `image_render` section and intervention checkpoint format are unchanged. Update class imports and names; existing rendering configs and weight files remain usable.

| Previous interface | Current location or replacement |
|---|---|
| Top-level training types or `vlmbox.training` | `vlmbox.trainer`: `Trainer`, `Sample`, `TrainStep`, `LossFunction` |
| Top-level generation result/error types | `vlmbox.generation` |
| `vlmbox.representation` helpers | `vlmbox.utils`: `mean_difference`, `normalize_direction`, `project`, `projection_length`, `cosine_similarity` |
| `vlmbox.repro.set_seed` | `vlmbox.utils.set_seed`, also available at the package root |
| `projection_coefficient` | Computed inside `project`; explicit scalar formulas belong in experiment code when needed |
| `angle` | Derive from cosine similarity in experiment code |
| `resize_image`, `crop_image`, `place_patch` | Use Pillow directly |
| `to_rgb`, `anchored_bbox`, low-level typography/composition | Internal `renderer.layout` and `renderer.intervention`; use `Renderer` as the public entry point |
| `adapter.final_norm`, `adapter.lm_head` | Access the native model explicitly for research that requires these components |
| Required metadata/adapter arguments to `VLM(...)` | `VLM(model, processor)`; optional keyword metadata and custom adapter |

Removed modules and top-level aliases have no compatibility shims. The optimizer component now lives in `vlmbox.trainer`; update previous `from vlmbox.training import ...` imports to `from vlmbox.trainer import ...`. Rendering implementation modules are not extension contracts. The root package contains no private helper modules.

The trainer's callable parameter is `loss_function`, typed by the structural `LossFunction` protocol. It receives `(outputs, inputs, samples)` from one forward and returns a scalar minimization loss. The name `loss` is reserved for scalar records such as `TrainStep.loss`.
