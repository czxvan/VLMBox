# Architecture

VLMBox is a thin research layer over PyTorch and Hugging Face Transformers. Transformers remains responsible for model execution, chat templates, image processing, and autoregressive generation. VLMBox adds reusable interfaces for inspecting and manipulating model internals.

## Model loading and execution

`VLM.from_pretrained()` accepts a ModelScope model ID and performs the following steps:

1. Resolve the checkpoint with ModelScope's `snapshot_download()`.
2. Load the native processor with `AutoProcessor`.
3. Load the model with `AutoModelForMultimodalLM`.
4. Construct a `TransformersAdapter` for accessing common decoder components.
5. Expose preparation, forward, generation, capture, and intervention methods through `VLM`.

The current loader uses `dtype="auto"`, `device_map="auto"`, and `trust_remote_code=True`. It intentionally does not expose a collection of model-loading overrides. Only load model repositories whose remote code you trust.

## Package modules

| Module | Responsibility |
|---|---|
| `vlmbox.model` | Model loading, input preparation, forward/generation, activation access, and generation traces |
| `vlmbox.adapter` | Resolves common language-decoder layouts in Transformers models |
| `vlmbox.activation` | Context-managed activation capture and intervention hooks |
| `vlmbox.representation` | Tensor operations for direction extraction, projection, and similarity |
| `vlmbox.image` | PIL image conversion, resizing, cropping, and patch placement |
| `vlmbox.render` | Fixed-size-font dynamic-height instruction bands, native-host image composition, layout metadata, and differentiable tensor concatenation |

| `vlmbox.training.image` | Universal perturbation and patch parameterizations |
| `vlmbox.training.trainer` | Optimization loop for universal image parameters |
| `vlmbox.generation` | Architecture-independent generation trace data structures |
| `vlmbox.repro` | Random seeding and optional deterministic PyTorch settings |

## Decoder adapter boundary

`TransformersAdapter` exposes decoder layers, attention modules, MLP modules, final normalization, and the language-model head through a small protocol. It recognizes a bounded set of common Transformers module paths. It deliberately avoids recursively searching the entire multimodal model, which could select a vision encoder instead of the language decoder.

If a model uses an unsupported decoder layout, adapter construction fails rather than guessing. Extending support should add a well-grounded layout to the adapter and corresponding tests; model-specific behavior should not leak into unrelated core modules.

## Activation operations

`VLM.capture()` and `VLM.intervene()` install temporary forward hooks and return context managers. Hooks are removed when the context exits, including when an exception occurs.

- Components: `block`, `attention`, or `mlp`.
- Token positions: `all`, `last`, or explicit sequence indices.
- Interventions: `add`, `replace`, `project_out`, or `zero`.
- Scope: `first_forward` or `every_forward`.

`first_forward` edits each selected module only on its first invocation. `every_forward` reapplies the edit on each invocation, including autoregressive decode steps.

## Design boundaries

VLMBox focuses on model-facing research primitives. It does not provide dataset catalogs, experiment tracking, benchmark orchestration, evaluation services, or a replacement image-processing implementation. Native Transformers processors remain the source of truth for model-specific preprocessing.
