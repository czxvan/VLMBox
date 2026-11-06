from __future__ import annotations

import pytest
import torch
from PIL import Image, ImageFont

from vlmbox.image import anchored_bbox
from vlmbox.render import (
    InstructionStyle,
    compose_instruction_image,
    compose_instruction_tensor,
    render_instruction_band,
)
from vlmbox.training.image import UniversalLocalPerturbation, UniversalPatch


@pytest.fixture
def style() -> InstructionStyle:
    # Use the system's redistributable test font, never bundle Arial.
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    return InstructionStyle(
        font_path=font.path, font_size=20, margin=20, line_spacing=4, alignment="center"
    )


def test_dynamic_band_preserves_host_dimensions_and_all_words(style: InstructionStyle) -> None:
    host = Image.new("RGB", (320, 217), (18, 44, 92))
    short = compose_instruction_image(host, "Do the task.", style)
    long = compose_instruction_image(
        host, " ".join(["please", "perform", "the", "task"] * 35), style
    )
    assert short.image.width == long.image.width == host.width
    assert short.image.height < long.image.height
    assert short.host_bbox == (0, short.instruction_bbox[3], 320, 217 + short.instruction_bbox[3])
    assert short.image.crop(short.host_bbox).tobytes() == host.tobytes()
    assert long.image.crop(long.host_bbox).tobytes() == host.tobytes()
    assert short.font_size == long.font_size == 20
    assert len(short.font_sha256) == 64
    assert long.metadata()["renderer"] == "virp-1"
    assert "please" in " ".join(long.lines)
    assert "task" in " ".join(long.lines)


def test_long_unbroken_instruction_wraps_without_clipping(style: InstructionStyle) -> None:
    text = "abcdefghij" * 50
    band = render_instruction_band(text, 100, style)
    assert len(band.lines) > 1
    assert "".join(band.lines) == text
    font = ImageFont.truetype(str(style.font_path), style.font_size)
    assert all(font.getlength(line) <= 100 - 2 * style.margin for line in band.lines)


def test_explicit_newlines_and_bottom_position(style: InstructionStyle) -> None:
    host = Image.new("RGB", (250, 100), "blue")
    result = compose_instruction_image(host, "first\nsecond", style, position="bottom")
    assert result.lines == ("first", "second")
    assert result.host_bbox == (0, 0, 250, 100)
    assert result.instruction_bbox[1] == 100
    assert result.image.crop(result.host_bbox).tobytes() == host.tobytes()
    blank = compose_instruction_image(host, "first\nsecond", style, draw_text=False)
    assert blank.image.size == result.image.size


def test_renderer_rejects_missing_fonts_and_narrow_width(style: InstructionStyle) -> None:
    with pytest.raises(FileNotFoundError):
        render_instruction_band(
            "hi", 100, InstructionStyle(font_path="/missing/font.ttf", font_size=20)
        )
    with pytest.raises(ValueError, match="width"):
        render_instruction_band("hi", 40, style)
    with pytest.raises(ValueError, match="nonempty"):
        render_instruction_band(" ", 100, style)


def test_geometry_is_host_relative_not_canvas_relative() -> None:
    assert anchored_bbox((322, 214), (192, 192), "bottom_right") == (130, 22, 322, 214)
    assert anchored_bbox((322, 214), (64, 64), "bottom_right") == (258, 150, 322, 214)
    with pytest.raises(ValueError, match="fit"):
        anchored_bbox((200, 150), (192, 192), "bottom_right")


def test_variable_size_universal_patch_preserves_gradient() -> None:
    patch = UniversalPatch(None, (2, 3), position="bottom_right", init="zeros")
    with torch.no_grad():
        patch.patch.fill_(0.75)
    for height, width in [(8, 9), (12, 6)]:
        original = torch.full((3, height, width), 0.2)
        modified = patch.apply(original)
        assert modified.shape == original.shape
        assert torch.all(modified[:, -2:, -3:] == 0.75)
        assert torch.all(modified[:, :-2, :] == 0.2)
        modified.sum().backward()
    assert patch.patch.grad is not None
    assert torch.count_nonzero(patch.patch.grad) > 0


def test_local_perturbation_is_shared_and_uses_same_support() -> None:
    perturbation = UniversalLocalPerturbation((2, 3), epsilon=0.1, position="bottom_right")
    with torch.no_grad():
        perturbation.delta.fill_(0.05)
    for height, width in [(8, 9), (12, 6)]:
        original = torch.full((3, height, width), 0.25)
        modified = perturbation.apply(original)
        assert torch.allclose(modified[:, -2:, -3:], torch.full((3, 2, 3), 0.30))
        assert torch.equal(modified[:, :-2, :], original[:, :-2, :])
        modified.sum().backward()
    assert perturbation.delta.grad is not None
    with torch.no_grad():
        perturbation.delta.fill_(0.5)
    perturbation.project_()
    assert torch.allclose(perturbation.delta, torch.full_like(perturbation.delta, 0.1))


def test_differentiable_host_patch_and_instruction_composition(style: InstructionStyle) -> None:
    band = render_instruction_band("Do the task.", 80, style)
    arr = __import__("numpy").asarray(band.image).copy()
    band_tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255
    host = torch.full((3, 75, 80), 0.2)
    patch = UniversalPatch(None, (16, 16), position="bottom_right")
    attacked_host = patch.apply(host)
    composed = compose_instruction_tensor(attacked_host, band_tensor)
    assert tuple(composed.shape) == (3, 75 + band.image.height, 80)
    assert torch.equal(composed[:, : band.image.height, :], band_tensor)
    composed.square().sum().backward()
    assert patch.patch.grad is not None
    assert torch.count_nonzero(patch.patch.grad) > 0


def test_tensor_composition_rejects_mismatched_width() -> None:
    with pytest.raises(ValueError, match="widths"):
        compose_instruction_tensor(torch.zeros(3, 2, 6), torch.zeros(3, 2, 7))
