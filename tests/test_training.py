from __future__ import annotations

import pytest
import torch
from PIL import Image

from vlmbox.training import ImageSample
from vlmbox.training.image import UniversalPatch, UniversalPerturbation


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


def test_image_sample_requires_pil_host_image() -> None:
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        ImageSample(image=torch.zeros(3, 4, 4), prompt="Describe the image.")

    sample = ImageSample(
        image=Image.new("RGB", (8, 6), "white"),
        prompt="Describe the image.",
        image_instruction="Execute the instruction in the image.",
    )
    assert sample.image.size == (8, 6)
