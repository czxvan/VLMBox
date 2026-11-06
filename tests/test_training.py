from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vlmbox.training import ImageSample, ImageTrainer, UniversalPatch, UniversalPerturbation


def test_universal_perturbation_is_shared_and_projected() -> None:
    perturbation = UniversalPerturbation((4, 5), epsilon=0.2)
    with torch.no_grad():
        perturbation.delta.fill_(0.1)

    images = torch.full((2, 3, 4, 5), 0.25)
    attacked = perturbation.apply(images)
    torch.testing.assert_close(attacked - images, torch.full_like(images, 0.1))

    attacked.sum().backward()
    assert perturbation.delta.grad is not None
    assert torch.count_nonzero(perturbation.delta.grad) > 0

    with torch.no_grad():
        perturbation.delta.fill_(0.5)
    perturbation.project_()
    assert float(perturbation.delta.detach().abs().max()) == pytest.approx(0.2)


def test_universal_patch_is_shared_and_preserves_gradient() -> None:
    patch = UniversalPatch(
        canvas_size=(4, 5),
        size=(2, 2),
        position="bottom_right",
        init="zeros",
    )
    with torch.no_grad():
        patch.patch.fill_(0.7)
    images = torch.zeros((2, 3, 4, 5))
    attacked = patch.apply(images)

    assert torch.all(attacked[:, :, 2:, 3:] == 0.7)
    assert torch.count_nonzero(attacked[:, :, :2, :3]) == 0
    attacked.sum().backward()
    assert patch.patch.grad is not None
    assert torch.count_nonzero(patch.patch.grad) > 0

    with torch.no_grad():
        patch.patch.fill_(1.2)
    patch.project_()
    assert float(patch.patch.detach().min()) == pytest.approx(1.0)
    assert float(patch.patch.detach().max()) == pytest.approx(1.0)


def test_universal_patch_validates_canvas_and_position() -> None:
    with pytest.raises(ValueError, match="fit inside"):
        UniversalPatch((4, 5), (5, 2))
    with pytest.raises(ValueError, match="outside"):
        UniversalPatch((4, 5), (2, 2), position=(4, 3))


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        (torch.zeros(1, 4, 4), ValueError, "shape"),
        (torch.zeros(3, 4, 4, dtype=torch.int64), TypeError, "floating-point"),
        (torch.full((3, 4, 4), 1.1), ValueError, r"\[0, 1\]"),
    ],
)
def test_image_sample_validates_rgb_float_inputs(image, error, message) -> None:
    with pytest.raises(error, match=message):
        ImageSample(image=image, prompt="Describe the image.")


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[0.5, -0.3, 0.2]]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-2, -1))
        return self.projection(pooled).unsqueeze(1)


class _FakeVLM:
    def __init__(self) -> None:
        self.model = _FakeModel()

    def prepare_differentiable(self, conversations, *, add_generation_prompt=True):
        images = []
        for conversation in conversations:
            for message in conversation:
                for part in message.get("content", []):
                    if part.get("type") == "image":
                        images.append(part["image"])
        return {"images": torch.stack(images)}

    def forward(self, inputs, **_kwargs):
        return SimpleNamespace(logits=self.model(inputs["images"]))


def _objective(outputs, _inputs, _samples):
    return outputs.logits.square().mean()


def test_image_trainer_updates_one_shared_parameter_and_restores_model_state() -> None:
    vlm = _FakeVLM()
    vlm.model.train()
    sample_a = ImageSample(torch.full((3, 8, 8), 0.4), "Describe image A.")
    sample_b = ImageSample(torch.full((3, 8, 8), 0.6), "Describe image B.")

    for image_parameter in (
        UniversalPerturbation((8, 8), epsilon=0.1),
        UniversalPatch((8, 8), (3, 3), position="center", seed=7),
    ):
        before = image_parameter.parameter.detach().clone()
        trainer = ImageTrainer(
            vlm,
            image_parameter,
            _objective,
            learning_rate=0.05,
        )
        history = trainer.fit(
            [sample_a, sample_b],
            max_steps=2,
            batch_size=2,
            shuffle=False,
        )

        assert len(history) == 2
        assert all(step.gradient_norm > 0 for step in history)
        assert not torch.equal(before, image_parameter.parameter.detach())
        assert image_parameter.parameter.grad is None
        assert vlm.model.training is True
        assert all(parameter.requires_grad for parameter in vlm.model.parameters())
        assert all(parameter.grad is None for parameter in vlm.model.parameters())


def test_image_trainer_rejects_mixed_canvas_sizes() -> None:
    vlm = _FakeVLM()
    trainer = ImageTrainer(
        vlm,
        UniversalPerturbation((8, 8), epsilon=0.1),
        _objective,
    )
    samples = [
        ImageSample(torch.zeros(3, 8, 8), "A"),
        ImageSample(torch.zeros(3, 7, 8), "B"),
    ]
    with pytest.raises(ValueError, match="same canvas size"):
        trainer.fit(samples, max_steps=1)


def test_image_trainer_accepts_variable_size_hosts_for_local_parameter() -> None:
    vlm = _FakeVLM()
    patch = UniversalPatch(None, (2, 2), position="bottom_right")
    trainer = ImageTrainer(vlm, patch, _objective, learning_rate=0.05)
    samples = [
        ImageSample(torch.full((3, 8, 9), 0.4), "A"),
        ImageSample(torch.full((3, 7, 8), 0.6), "B"),
    ]
    before = patch.parameter.detach().clone()
    history = trainer.fit(samples, max_steps=2, batch_size=1, shuffle=False)
    assert len(history) == 2
    assert not torch.equal(before, patch.parameter.detach())
    assert all(step.gradient_norm > 0 for step in history)
