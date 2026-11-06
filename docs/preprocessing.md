# Image preprocessing and gradients

VLMBox delegates model-specific image processing to each model's native Transformers processor. It does not replace the processor's resize, crop, patchification, image-splitting, or normalization logic.

## Two preparation paths

### `VLM.prepare()`

Use `prepare()` for ordinary inference with the image and content types accepted by the model's native chat template and processor. A typical message is:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Describe the image."},
        ],
    }
]
inputs = vlm.prepare(messages)
```

It accepts either one conversation (`list[message]`) or a batch (`list[list[message]]`).

### `VLM.prepare_differentiable()`

Use this path when image content is a trainable PyTorch Tensor and gradients must flow from the model output back to the image or universal image parameter.

Each image item must contain an RGB `torch.Tensor` with shape `[3, height, width]`, a floating-point dtype, finite values, and values in `[0, 1]`. A batch is represented as a sequence of conversations, each containing at least one Tensor image.

```python
image = torch.rand(3, 224, 224, requires_grad=True)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the image."},
        ],
    }
]
inputs = vlm.prepare_differentiable(messages)
outputs = vlm.forward(inputs)
loss = outputs.logits.float().square().mean()
loss.backward()
assert image.grad is not None
```

The differentiable path converts image tensors to contiguous CPU `float32` tensors before calling the native processor. It passes `do_rescale=False` because the input contract already requires `[0, 1]` values. After preprocessing, the resulting model inputs are moved to `vlm.input_device`. These transfers remain in the autograd graph; the input image is not detached by preparation.

## Device policy

Preprocessing intentionally runs on CPU. This preserves the resampling behavior configured by the native processor, including kernels that may not be available on CUDA. VLMBox does not expose a preprocessing-device switch or silently retry preprocessing on another device.

A practical setup is to keep source images and universal image parameters on CPU while the VLM runs on the accelerator. The differentiable transfer from the image parameter through CPU preprocessing and back to the model device preserves gradients, although transfers have a performance cost.

Do not assume CPU and CUDA preprocessing produce identical `pixel_values` for every processor, model, library version, or image. VLMBox standardizes its own differentiable path on CPU; it does not promise numerical equivalence with independently constructed GPU preprocessing pipelines.

## Batching and chat templates

Both preparation methods enable padding and configure left padding for decoder-only autoregressive generation. The `VLM` wrapper sets the processor tokenizer's `padding_side` to `left` when that attribute is available.

Model-specific chat-template variables can be passed without replacing the native template:

```python
inputs = vlm.prepare(
    messages,
    chat_template_kwargs={"enable_thinking": False},
)
```

For supervised target preparation, both methods accept `add_generation_prompt=False` when the conversation already contains the assistant target.

## What is and is not guaranteed

VLMBox guarantees the input validation and device-transfer behavior described above. The exact resized pixels, crop layout, patch count, image-grid metadata, and normalization are determined by the model's native processor and its installed dependencies. For reproducible experiments, record the model revision and relevant PyTorch, Transformers, and image-processing dependency versions alongside the preprocessing configuration.
