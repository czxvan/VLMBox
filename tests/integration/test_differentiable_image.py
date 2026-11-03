from __future__ import annotations

import gc
import os

import pytest
import torch
import torch.nn.functional as F

from vlmbox import VLM, ImageSample, ImageTrainer, UniversalPatch, UniversalPerturbation

MODEL_IDS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "OpenGVLab/InternVL3_5-2B-HF",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
)
IMAGE_SIZES = ((224, 224), (192, 320))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("VLMBOX_RUN_GPU_INTEGRATION") != "1",
        reason="set VLMBOX_RUN_GPU_INTEGRATION=1 to load cached VLM checkpoints",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _conversation(image: torch.Tensor, target: str | None = None) -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one word."},
            ],
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def _assert_native_processor_parity(vlm: VLM, image: torch.Tensor) -> None:
    actual = vlm.prepare_differentiable(_conversation(image))
    expected = vlm.processor.apply_chat_template(
        _conversation(image.detach().clone()),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_rescale": False},
    )

    expected_keys = {key for key, value in expected.items() if isinstance(value, torch.Tensor)}
    actual_keys = {key for key, value in actual.items() if isinstance(value, torch.Tensor)}
    assert actual_keys == expected_keys

    for key in sorted(expected_keys):
        actual_value = actual[key].detach().cpu()
        expected_value = expected[key].detach().cpu()
        assert actual_value.shape == expected_value.shape, key
        assert actual_value.dtype == expected_value.dtype, key
        if actual_value.is_floating_point():
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0, msg=key)
        else:
            assert torch.equal(actual_value, expected_value), key

    assert actual["pixel_values"].device == vlm.input_device
    gradient = torch.autograd.grad(actual["pixel_values"].float().sum(), image)[0]
    assert bool(torch.isfinite(gradient).all().item())
    assert float(gradient.abs().sum().item()) > 0.0


def _target_prefix_length(vlm: VLM, image: torch.Tensor) -> int:
    prompt_inputs = vlm.prepare_differentiable(_conversation(image))
    target_inputs = vlm.prepare_differentiable(
        _conversation(image, target="HACKED"),
        add_generation_prompt=False,
    )
    prompt_ids = prompt_inputs["input_ids"][0].detach().cpu()
    target_ids = target_inputs["input_ids"][0].detach().cpu()
    prefix_length = prompt_ids.numel()
    assert target_ids.numel() > prefix_length
    assert torch.equal(target_ids[:prefix_length], prompt_ids)
    return prefix_length


def _target_ce_objective(prefix_length: int):
    def objective(outputs, inputs, _samples) -> torch.Tensor:
        labels = inputs["input_ids"][:, prefix_length:]
        mask = inputs["attention_mask"][:, prefix_length:].bool()
        logits = outputs.logits[:, prefix_length - 1 : -1, :].float()
        assert logits.shape[:2] == labels.shape
        assert bool(mask.any().item())
        return F.cross_entropy(logits[mask], labels[mask])

    return objective


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_native_processor_parity_and_universal_training_gradients(model_id: str) -> None:
    vlm = VLM.from_pretrained(model_id, local_files_only=True)
    assert vlm.input_device.type == "cuda"

    for parameter in vlm.model.parameters():
        parameter.requires_grad_(False)

    if "SmolVLM" in model_id:
        assert vlm.processor.image_processor.do_image_splitting is True

    for height, width in IMAGE_SIZES:
        image = torch.rand(3, height, width, dtype=torch.float32, requires_grad=True)
        _assert_native_processor_parity(vlm, image)
        del image
        gc.collect()

    image = torch.rand(3, 224, 224, dtype=torch.float32, requires_grad=True)
    inputs = vlm.prepare_differentiable(_conversation(image))
    outputs = vlm.forward(inputs, use_cache=False)
    loss = outputs.logits[:, -1, :].float().square().mean()
    loss.backward()
    assert image.grad is not None
    assert bool(torch.isfinite(image.grad).all().item())
    assert float(image.grad.abs().sum().item()) > 0.0
    assert all(parameter.grad is None for parameter in vlm.model.parameters())

    del outputs, inputs, loss, image
    gc.collect()
    torch.cuda.empty_cache()

    samples = [
        ImageSample(
            image=torch.rand(3, 224, 224),
            prompt="Describe this image in one word.",
            target="HACKED",
        )
        for _ in range(2)
    ]
    prefix_length = _target_prefix_length(vlm, samples[0].image)
    parameterizations = (
        UniversalPerturbation((224, 224), epsilon=8 / 255),
        UniversalPatch(
            (224, 224),
            (32, 32),
            position="bottom_right",
            seed=17,
        ),
    )
    for image_parameter in parameterizations:
        assert next(image_parameter.parameters()).device.type == "cpu"
        before = image_parameter.parameter.detach().clone()
        trainer = ImageTrainer(
            vlm,
            image_parameter,
            _target_ce_objective(prefix_length),
            learning_rate=1e-3,
        )
        history = trainer.fit(samples, max_steps=1, batch_size=2, shuffle=False)
        assert history[0].gradient_norm > 0.0
        assert not torch.equal(before, image_parameter.parameter.detach())
        assert all(parameter.grad is None for parameter in vlm.model.parameters())
        del trainer, history
        gc.collect()
        torch.cuda.empty_cache()

    del vlm
    gc.collect()
    torch.cuda.empty_cache()
