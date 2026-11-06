from __future__ import annotations

import pytest
import torch
from PIL import Image

from vlmbox.renderer.intervention import Intervention
from vlmbox.trainer import Sample


def test_universal_perturbation_is_shared_and_projected() -> None:
    perturbation = Intervention("local_additive", (2, 3), epsilon=0.2)
    with torch.no_grad():
        perturbation.parameter.fill_(0.1)

    images = torch.full((2, 3, 4, 5), 0.25)
    attacked = perturbation(images)
    expected = torch.zeros_like(images)
    expected[:, :, -2:, -3:] = 0.1
    torch.testing.assert_close(attacked - images, expected)

    attacked.sum().backward()
    assert perturbation.parameter.grad is not None
    assert torch.count_nonzero(perturbation.parameter.grad) > 0

    with torch.no_grad():
        perturbation.parameter.fill_(0.5)
    perturbation.project_()
    assert float(perturbation.parameter.detach().abs().max()) == pytest.approx(0.2)


def test_universal_patch_is_shared_and_preserves_gradient() -> None:
    patch = Intervention(
        "overlay_patch",
        size=(2, 2),
        anchor="bottom_right",
        init="zeros",
    )
    with torch.no_grad():
        patch.parameter.fill_(0.7)
    images = torch.zeros((2, 3, 4, 5))
    attacked = patch(images)

    assert torch.all(attacked[:, :, 2:, 3:] == 0.7)
    assert torch.count_nonzero(attacked[:, :, :2, :3]) == 0
    attacked.sum().backward()
    assert patch.parameter.grad is not None
    assert torch.count_nonzero(patch.parameter.grad) > 0

    with torch.no_grad():
        patch.parameter.fill_(1.2)
    patch.project_()
    assert float(patch.parameter.detach().min()) == pytest.approx(1.0)
    assert float(patch.parameter.detach().max()) == pytest.approx(1.0)


def test_universal_patch_validates_host_and_position() -> None:
    with pytest.raises(ValueError, match="fit inside"):
        Intervention("overlay_patch", (5, 2))(torch.zeros(3, 4, 5))
    with pytest.raises(ValueError, match="outside"):
        Intervention("overlay_patch", (2, 2), anchor=(4, 3))(torch.zeros(3, 4, 5))


def test_image_sample_requires_pil_host_image() -> None:
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        Sample(image=torch.zeros(3, 4, 4), prompt="Describe the image.")

    sample = Sample(
        image=Image.new("RGB", (8, 6), "white"),
        prompt="Describe the image.",
        image_instruction="Execute the instruction in the image.",
    )
    assert sample.image.size == (8, 6)
