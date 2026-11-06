# Usage

Run examples from the repository root after the [installation](../README.md#install). The Python sections below build on the model, image, and inputs created in the first section. Model execution needs hardware suitable for the checkpoint; rendering alone needs no model.

## Load a model and prepare inputs

```python
import torch
from PIL import Image
from vlmbox import VLM, set_seed

set_seed(42)
vlm = VLM.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
image = Image.new("RGB", (224, 224), "red")


def conversation(image, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


messages = conversation(image, "Describe the image briefly.")
inputs = vlm.prepare(messages)
sequences = vlm.generate(inputs, max_new_tokens=16, do_sample=False)
new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
print(vlm.processor.batch_decode(new_tokens, skip_special_tokens=True))
```

`from_pretrained` accepts a ModelScope ID. Use `local_files_only=True` for cached models. To supply your own loaded model and processor, use `VLM(model, processor)`; unsupported decoder layouts can provide `adapter=`.

Batch complete conversations, rather than individual messages:

```python
batch = [messages, conversation(image, "Name the dominant color.")]
batch_inputs = vlm.prepare(batch)
batch_tokens = vlm.generate(batch_inputs, max_new_tokens=8, do_sample=False)
answers = batch_tokens[:, batch_inputs["input_ids"].shape[1] :]
print(vlm.processor.batch_decode(answers, skip_special_tokens=True))
```

Both preparation paths use left padding. Pass model-specific template variables with `chat_template_kwargs={"enable_thinking": False}` when supported by the model. A conversation already containing assistant target text uses `add_generation_prompt=False`; continuing an assistant prefix additionally uses `chat_template_kwargs={"continue_final_message": True}`.

## Capture and edit decoder activations

```python
layer = len(vlm.adapter.layers) // 2
with torch.no_grad(), vlm.capture("block", layers=layer, positions="last") as states:
    vlm.forward(inputs, use_cache=False)
print(states[layer].shape)  # [batch, 1, hidden]

# Ablate the selected block's final prompt position on its first invocation.
with vlm.intervene("block", layer, "zero", positions="last", scope="first_forward"):
    ablated_tokens = vlm.generate(inputs, max_new_tokens=8, do_sample=False)
```

All layers are zero-based decoder blocks; states precede separate final model normalization. Capture records the first invocation. Use `scope="every_forward"` for edits throughout decoding. Other operations are `add`, `replace`, and `project_out`; they take a Tensor `value` broadcastable to the selected activations. Hooks are removed when the context exits, including on failure. Enter an intervention before capture to observe edited states.

## Trace generation

```python
trace = vlm.generate_trace(
    inputs,
    capture_logits=True,
    capture_hidden_states=True,
    hidden_state_layers=[layer],
    decode_text=True,
    max_new_tokens=4,
    do_sample=False,
)
print(trace.texts)
print(trace.generated_token_ids.shape)
if trace.steps:
    step = trace.steps[0]
    print(step.phase, step.token_ids, step.hidden_states[layer].shape)
```

Step zero is the prompt forward that predicts the first generated token. Later steps record forwards predicting subsequent tokens. Capture requires decoder-only generation, `num_beams=1`, and one selected-block forward per generated token. Select only the blocks you need: logits and states retain device memory. Finish reasons are estimates.

## Render an image

Copy the [YAML](../examples/image_render.yaml):

```bash
mkdir -p configs
cp examples/image_render.yaml configs/image_render.yaml
```

Set `instruction.font.path` to an existing licensed font file; relative paths resolve from the YAML file. Set `intervention.size` to `[32, 32]` for this 224-pixel host. Each intervention must fit inside every host image.

```python
from vlmbox import Renderer

renderer = Renderer.from_yaml("configs/image_render.yaml")
preview, metadata = renderer.render(image, "Describe the red image.", return_metadata=True)
preview.save("preview.png")
print(metadata["host_bbox"], metadata["intervention_bbox"])

pixels = renderer(image, "Describe the red image.")
print(pixels.shape)  # RGB CHW Tensor in [0, 1], connected to renderer parameters
```

The YAML selects `overlay_patch`, bounded `local_additive`, or `none`. `.render(...)` produces PIL; calling the module produces a differentiable Tensor. Hosts retain their size, and the text band adds height. See the [rendering protocol](protocols/visual-input-rendering.md) for geometry and controls.

## Train with a loss function

This minimal objective encourages the **first token** of `red` at the next response position. It demonstrates the loss/gradient interface; it does not train a complete target sequence.

```python
import torch.nn.functional as F
from vlmbox.trainer import Sample, Trainer

target_id = vlm.processor.tokenizer.encode("red", add_special_tokens=False)[0]


def loss_function(outputs, inputs, samples):
    logits = outputs.logits[:, -1, :].float()
    labels = torch.full((len(samples),), target_id, device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


samples = [
    Sample(
        image=image,
        prompt="Name the dominant color.",
        image_instruction="Describe the red image.",
    )
]
trainer = Trainer(vlm, renderer, loss_function=loss_function, learning_rate=1e-3)
history = trainer.fit(samples, max_steps=3, batch_size=1, seed=42)
print(history[-1])  # TrainStep(step, loss, gradient_norm)
renderer.save_intervention("intervention.pt")
```

A `LossFunction` receives `(outputs, inputs, samples)` in the same batch order and returns one finite scalar Tensor to minimize. Ordinary functions and callable objects work without inheritance. `inputs` contains processed token IDs/masks; `samples` contains original prompts and targets. `Trainer` freezes the VLM, optimizes renderer parameters, projects their bounds, and restores model parameter flags and training modes even on failure.

Reuse weights with a compatible YAML configuration:

```python
restored = Renderer.from_yaml("configs/image_render.yaml")
restored.load_intervention("intervention.pt")
restored.render(image, "Describe the red image.").save("restored.png")
```

Checkpoints contain intervention weights and geometry, not optimizer state. Loading requires matching type, size, anchor, and epsilon.

## Custom differentiable loops

Use `renderer.parameters()` with a PyTorch optimizer and call `renderer.project_()` after each update. For losses needing hooks or multiple model forwards, own the loop and prepare Tensor images directly:

```python
vlm.model.eval().requires_grad_(False)
optimizer = torch.optim.Adam(renderer.parameters(), lr=1e-3)
optimizer.zero_grad(set_to_none=True)
pixels = renderer(image, "Describe the red image.")
inputs_with_grad = vlm.prepare_differentiable(conversation(pixels, "Name the dominant color."))
outputs = vlm.forward(inputs_with_grad, use_cache=False)
loss = loss_function(outputs, inputs_with_grad, samples)
loss.backward()
assert any(parameter.grad is not None for parameter in renderer.parameters())
optimizer.step()
renderer.project_()
```

Tensor images must be finite floating-point RGB CHW values in `[0, 1]`, with at least one Tensor image per conversation. Native preprocessing runs on CPU float32 with `do_rescale=False`, then inputs move to the model device. Both transfers retain gradients. `forward()` keeps autograd enabled; avoid `inference_mode()` or detaching image tensors during optimization.

## Runnable experiments

With images under `data/images/` and the configured font available, optimize a complete teacher-forced assistant target:

```bash
python examples/universal_image_optimization.py \
  --model-id HuggingFaceTB/SmolVLM-500M-Instruct \
  --image-dir data/images --render-config configs/image_render.yaml \
  --target RED --steps 100 --batch-size 2 --output outputs/preview.png
```

This writes `preview.png`, `preview.pt`, and `preview.json`. The script derives assistant-token spans, handles padding and causal label shifts, and rejects templates whose prompt prefix is inconsistent. Pass `--intervention outputs/preview.pt` to initialize a new run from learned weights.

For a paired-prompt direction and activation-steering probe, using a synthetic image by default:

```bash
python examples/paired_prompt_steering.py \
  --model-id HuggingFaceTB/SmolVLM-500M-Instruct \
  --max-new-tokens 4 --output outputs/steering.json
```

Both scripts accept `--local-files-only` and `--help`; the steering script requires CUDA. Experiment code owns datasets, loss semantics, token masks, metrics, and reporting; VLMBox supplies shared model and rendering operations.
