from __future__ import annotations

import math

import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from vlmbox.renderer.intervention import Intervention, anchored_bbox
from vlmbox.renderer.layout import (
    InstructionStyle,
    compose_instruction_image,
    render_instruction_band,
    to_rgb,
)


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
    assert long.metadata()["renderer"] == "virp-2"
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
    patch = Intervention("overlay_patch", (2, 3), anchor="bottom_right", init="zeros")
    with torch.no_grad():
        patch.parameter.fill_(0.75)
    for height, width in [(8, 9), (12, 6)]:
        original = torch.full((3, height, width), 0.2)
        modified = patch(original)
        assert modified.shape == original.shape
        assert torch.all(modified[:, -2:, -3:] == 0.75)
        assert torch.all(modified[:, :-2, :] == 0.2)
        modified.sum().backward()
    assert patch.parameter.grad is not None
    assert torch.count_nonzero(patch.parameter.grad) > 0


def test_local_perturbation_is_shared_and_uses_same_support() -> None:
    perturbation = Intervention("local_additive", (2, 3), epsilon=0.1, anchor="bottom_right")
    with torch.no_grad():
        perturbation.parameter.fill_(0.05)
    for height, width in [(8, 9), (12, 6)]:
        original = torch.full((3, height, width), 0.25)
        modified = perturbation(original)
        assert torch.allclose(modified[:, -2:, -3:], torch.full((3, 2, 3), 0.30))
        assert torch.equal(modified[:, :-2, :], original[:, :-2, :])
        modified.sum().backward()
    assert perturbation.parameter.grad is not None
    with torch.no_grad():
        perturbation.parameter.fill_(0.5)
    perturbation.project_()
    assert torch.allclose(perturbation.parameter, torch.full_like(perturbation.parameter, 0.1))


@pytest.mark.parametrize("font_name", ["DejaVuSans.ttf", "DejaVuSans-Oblique.ttf"])
@pytest.mark.parametrize("alignment", ["left", "center", "right"])
def test_zero_margin_text_preserves_all_glyph_pixels(font_name, alignment) -> None:
    font = ImageFont.truetype(font_name, 20)
    text = "jgypÁ"
    reference = Image.new("RGB", (150, 100), "white")
    ImageDraw.Draw(reference).text((20, 20), text, font=font, fill="black", anchor="lt")
    ink = Image.eval(reference, lambda channel: 255 - channel)
    expected = reference.crop(ink.getbbox())
    width = max(font.getbbox(text)[2], font.getlength(text)) - min(0, font.getbbox(text)[0])
    band = render_instruction_band(
        text,
        math.ceil(width),
        InstructionStyle(font.path, 20, alignment=alignment),
    )
    assert band.lines == (text,)
    ink = Image.eval(band.image, lambda channel: 255 - channel)
    actual = band.image.crop(ink.getbbox())
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


def test_to_rgb_converts_mode_and_returns_independent_copy() -> None:
    source = Image.new("RGBA", (3, 2), (10, 20, 30, 40))

    result = to_rgb(source)

    assert result.mode == "RGB"
    assert result.size == source.size
    assert result.getpixel((0, 0)) == (10, 20, 30)
    result.putpixel((0, 0), (255, 0, 0))
    assert source.getpixel((0, 0)) == (10, 20, 30, 40)


def test_to_rgb_copies_existing_rgb_image() -> None:
    source = Image.new("RGB", (2, 2), (1, 2, 3))

    result = to_rgb(source)
    result.putpixel((0, 0), (9, 9, 9))

    assert result is not source
    assert source.getpixel((0, 0)) == (1, 2, 3)


def test_to_rgb_rejects_non_pillow_input() -> None:
    with pytest.raises(TypeError, match="PIL.Image"):
        to_rgb("image.png")  # type: ignore[arg-type]
